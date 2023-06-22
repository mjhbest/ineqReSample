import numpy as np
import pandas as pd
import warnings
import random
import collections
from scipy.stats import gini

def gini(x, weights = np.ones(len(x))):
    ox = np.argsort(x)
    x = x[ox]
    weights = weights[ox]/sum(weights)
    p = np.cumsum(weights)
    nu = np.cumsum(weights * x)
    n = len(nu)
    nu = nu/nu[n-1]
    return sum(nu[:-1] * p[1:]) - sum(nu[1:] * p[:-1])


def citeIneq(result, metric="all", type="resampled", showMargins=False):
    """
    Takes an adjustCiteMetrics object and returns a dataframe of 
    inequality metrics.
    
    Parameters
    ----------
    result : adjustCiteMetrics object
        An adjustCiteMetrics object created by the adjustCiteMetrics
        function.
    metric : list
        a list of metrics to calculate. If "all" (default), all available metrics
        will be calculated.
    type : {"resampled", "uncorrected"}
        "resampled" (default) calculates inequality metrics using the 
        resampled data. "uncorrected" calculates inequality metrics using 
        the uncorrected data. Can also be a list of both.
    showMargins : bool
        If True, add margins to the output dataframe. Default is False.
    
    Returns
    -------
    DataFrame
        A dataframe of inequality metrics.
    """
    
    # Check that result is an adjustCiteMetrics class
    if not isinstance(result, adjustCiteMetrics):
        raise ValueError("result must be of class adjustCiteMetrics")
    
    # Check for possible type
    if (type != "resampled") and (type != "uncorrected"):
        raise ValueError("type must be `resampled' or `uncorrected'")
    
    # Check for all metrics
    if metric == "all":
        metric = result.availMetrics
    
    # Check called metrics exist in adjustCiteMetrics class
    if set(metric) - set(result.availMetrics):
        raise ValueError("metric can only request inequality measures reported in the supplied adjustCiteMetrics object")
    
    # Combination of metric and type
    if "resampled" in type:
        metricRS = [m+"RS" for m in metric]
    else:
        metricRS = None
    
    if "uncorrected" in type:
        metricUC = [m+"UC" for m in metric]
    else:
        metricUC = None
    
    # Apply margins option
    if showMargins:
        marginsList = ["nR", "nR0", "nSXnC", "nS0XnC0"]
    else:
        marginsList = None
    
    metric = ["year"] + metricRS + metricUC + marginsList
    
    result = result[metric]
    result = result.sort_values(by="year")
    result.index = range(1, len(result)+1)
    
    return result


def adjustCiteMetrics(papers, cites, pubID="paperID", pubYear="publishedYear", citedID="citedPaperID", citedYear="citedPaperYear", citingYear=None, citationWindow=None, quantiles=[0.2, 0.8], refYear=None, refPaperCount=None, refCiteCount=None, sims=10, lowCitesInclusion=False, lowCitesThreshold=0.1, paperDupThreshold=0.95, periods=None, verbose=False):
    if not isinstance(papers, pd.DataFrame):
        raise ValueError("papers must be a data frame with columns containing the publication ID and publication year")
    if not isinstance(cites, pd.DataFrame):
        raise ValueError("cites must be a data frame with columns containing the IDs of cited papers and years in which they were cited by other papers")
    if pubID not in papers.columns:
        raise ValueError(f"papers does not contain a column named {pubID}: pubID should refer to a variable in papers")
    if pubYear not in papers.columns:
        raise ValueError(f"papers does not contain a column named {pubYear}: pubYear should refer to a variable in papers")
    if citedID not in cites.columns:
        raise ValueError(f"cites does not contain a column named {citedID}: citedID should refer to a variable in cites")
    if citedYear not in cites.columns:
        raise ValueError(f"cites does not contain a column named {citedYear}: citedYear should refer to a variable in cites")
    if citingYear is not None and citingYear not in cites.columns:
        raise ValueError(f"cites does not contain a column named {citingYear}: citingYear should refer to a variable in cites or be left None")
    if citingYear is not None and citationWindow is None:
        raise ValueError("Citing paper's published year needs to be provided to apply the citation window")
    if citationWindow is not None and not isinstance(citationWindow, (int, float)):
        raise ValueError("citationWindow must be a numeric scalar or None (the default)")
    sims = int(sims)
    if sims <= 0:
        raise ValueError("sims must be a positive integer")
    if not isinstance(paperDupThreshold, (int, float)) or paperDupThreshold <= 0 or paperDupThreshold > 1:
        raise ValueError("paperDupThreshold should be a numeric scalar in the interval (0,1]")
    if not isinstance(quantiles, list) or any(q >= 1 or q <= 0 for q in quantiles):
        raise ValueError("quantiles should be numeric in the interval (0,1)")
    quantiles = [q * 100 for q in quantiles]
    qInt = [int(q) for q in quantiles]
    qDec = [q % 1 for q in quantiles]
    qStr = ['q' + str(q).zfill(2) for q in qInt]
    for i in range(len(qStr)):
        if qDec[i] > 0:
            qStr[i] = "q" + str(quantiles[i]).rstrip('0').rstrip('.')
    if refYear is not None and not isinstance(refYear, (int, float)):
        raise ValueError("refYear should be a numeric scalar identifying the referenced year among available pubYear in papers")
    if refPaperCount is not None and (not isinstance(refPaperCount, (int, float)) or refPaperCount <= 0):
        raise ValueError("refPaperCount must be None or a numeric scalar identifying the referenced publication count")
    if refCiteCount is not None and (not isinstance(refCiteCount, (int, float)) or refCiteCount <= 0):
        raise ValueError("refCiteCount must be None or a positive numeric scalar identifying the referenced citation count")
    if not isinstance(lowCitesInclusion, bool):
        raise ValueError("lowCitesInclusion must be logical")
    if not isinstance(lowCitesThreshold, (int, float)) or lowCitesThreshold < 0 or lowCitesThreshold > 1:
        raise ValueError("lowCitesThreshold should be numeric in the interval [0,1]")
    if periods is not None and not isinstance(periods, (list, np.ndarray)):
        raise ValueError("periods should be numeric vector or None (the default)")
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be logical")
    
    if pubID in papers.columns:
        papers['paperID'] = papers[pubID].astype(str)
    else:
        raise ValueError(f"papers does not contain a column named {pubID}: pubID should refer to a variable in papers")
    if pubYear in papers.columns:
        papers['publishedYear'] = papers[pubYear]
    else:
        raise ValueError(f"papers does not contain a column named {pubYear}: pubYear should refer to a variable in papers")
    
    if citedID in cites.columns:
        cites['citedPaperID'] = cites[citedID].astype(str)
    else:
        raise ValueError(f"cites does not contain a column named {citedID}: citedID should refer to a variable in cites")
    if citedYear in cites.columns:
        cites['citedPaperYear'] = cites[citedYear]
    else:
        raise ValueError(f"cites does not contain a column named {citedYear}: citedYear should refer to a variable in cites")
    if citingYear is None:
        cites['citingPaperYear'] = cites[citingYear]
    elif citingYear in cites.columns:
        cites['citingPaperYear'] = cites[citingYear]
    else:
        raise ValueError(f"cites does not contain a column named {citingYear}: citingYear should refer to a variable in cites or be left None")
    
    citesBeforeFilterLength = len(cites)
    cites = cites[cites['citedPaperID'].isin(papers['paperID'])]
    citesAfterFilterLength = len(cites)
    if citesBeforeFilterLength != citesAfterFilterLength:
        print("Some citation information in the cites dataframe have been deleted because the cited papers do not exist in papers")
    
    if cites['citingPaperYear'].isnull().all() and citationWindow is not None:
        raise ValueError("Citing paper's published year needs to be provided to apply the citation window")
    elif not cites['citingPaperYear'].isnull().all():
        cites = cites[(cites['citingPaperYear'] - cites['citedPaperYear']) <= citationWindow]
    
    if refYear is not None:
        if refYear not in papers['publishedYear'].unique():
            raise ValueError("refYear must be included in pubYears in papers")
        if refPaperCount is not None:
            raise ValueError("either refYear or refPaperCount must be provided.")
        if refCiteCount is not None:
            raise ValueError("either refYear or refCiteCount must be provided.")
        nR0 = len(papers[papers['publishedYear'] == refYear])
        nS0XnC0 = len(cites[cites['citedPaperYear'] == refYear])
    if refYear is None:
        if refPaperCount is None:
            raise ValueError("Either refYear or refPaperCount needs to be provided.")
        if refCiteCount is None:
            raise ValueError("Either refYear or refCiteCount needs to be provided.")
        nR0 = refPaperCount
        nS0XnC0 = refCiteCount
    
    if periods is None:
        years = papers['publishedYear'].unique()
    elif np.isin(periods, papers['publishedYear']).all():
        years = periods
    else:
        raise ValueError("periods should exist in papers$pubYear")
    
    nR0 = int(nR0)
    nS0XnC0 = int(nS0XnC0)
    resALL = pd.DataFrame(columns=['year', 'giniUC', 'everCitedUC', 'hhiUC', 'giniRS', 'everCitedRS', 'hhiRS', 'nR', 'nR0', 'nSXnC', 'nS0XnC0', 'lowCites'])
    for year in years:
        if refYear is None:
            reference = False
        elif year == refYear:
            reference = True
        else:
            reference = False
        
        papersYear = papers[papers['publishedYear'] == year]['paperID']
        citesYear = cites[cites['citedPaperYear'] == year]['citedPaperID']
        
        nR = len(papersYear.unique())
        nSXnC = len(citesYear)
        
        cumDistRS = np.zeros(nR0)
        
        giniCrs = np.zeros(sims)
        herfRS = np.zeros(sims)
        withCitesRS = np.zeros(sims)
        pctListRS = pd.DataFrame(np.zeros((sims, len(quantiles))), columns=[f"pct{int(q)}RS" for q in quantiles])
        lowCites = np.zeros(sims)
        
        for q in range(len(quantiles)):
            if qDec[q] > 0:
                qStr[q] = f"q{quantiles[q]}"
        
        for j in range(sims):
            if not reference and nR0 > (paperDupThreshold * nR):
                duplicated = True
                papersDup = papersYear.copy()
                citesDup = citesYear.copy()
                ctr = 0
                done = False
                while not done:
                    ctr += 1
                    papers0 = papersYear + "dup" + str(ctr)
                    papersDup = pd.concat([papersDup, papers0])
                    cites0 = citesYear + "dup" + str(ctr)
                    citesDup = pd.concat([citesDup, cites0])
                    if len(papersDup) > (nR0 * (1 + paperDupThreshold)):
                        done = True
                papersRS = papersDup.sample(nR0, replace=False)
            else:
                duplicated = False
                papersRS = papersYear.sample(nR0, replace=False)
            
            citeDistRS = np.zeros(nR0)
            
            if duplicated:
                eligibleCites = citesDup[citesDup.isin(papersRS)]
            else:
                eligibleCites = citesYear[citesYear.isin(papersRS)]
            
            if nS0XnC0 <= len(eligibleCites):
                citedPapersRS = eligibleCites.sample(nS0XnC0, replace=False)
            else:
                if verbose:
                    if lowCitesInclusion:
                        warnMsg = "Using all available cites."
                    else:
                        warnMsg = f"Year {year} reported as NA for corrected inequality metrics."
                    print(f"Not enough citations for complete subsample in year {year}.  Needed {nS0XnC0}; had {len(eligibleCites)}. {warnMsg}")
                citedPapersRS = eligibleCites
                lowCites[j] = 1
            
            citeCountsRS = np.zeros(nR0)
            citeCountsRS0 = np.sort(citedPapersRS.value_counts().values)[::-1]
            citeCountsRS[:len(citeCountsRS0)] = citeCountsRS0
            cumDistRS += citeCountsRS
            
            giniCrs[j] = gini(citeCountsRS)
            
            withCitesRS[j] = np.sum(citeCountsRS > 0) / nR0
            
            cumpctRS = 100 * np.cumsum(citeCountsRS) / np.sum(citeCountsRS)
            for q in range(len(quantiles)):
                pctListRS.iloc[j, q] = 100 * (np.sum(cumpctRS < quantiles[q]) + 1) / nR0
        
            sharesRS = citeCountsRS / np.sum(citeCountsRS)
            herfRS[j] = np.sum(sharesRS ** 2)
        
        citeCounts = np.zeros(nR)
        citeCounts0 = np.sort(citesYear.value_counts().values)[::-1]
        citeCounts[:len(citeCounts0)] = citeCounts0
        
        giniC = gini(citeCounts)
        
        withCites = np.sum(citeCounts > 0) / nR
        
        cumpct = 100 * np.cumsum(citeCounts) / np.sum(citeCounts)
        pctList = pd.DataFrame(100 * (np.sum(cumpct < quantiles[:, np.newaxis], axis=1) + 1) / nR, columns=[f"pct{int(q)}" for q in quantiles])
        
        shares = citeCounts / np.sum(citeCounts)
        herf = np.sum(shares ** 2)
        
        lowCites = np.mean(lowCites)
        
        res = {'giniUC': giniC,
               'everCitedUC': withCites,
               'hhiUC': herf,
               'giniRS': np.mean(giniCrs),
               'everCitedRS': np.mean(withCitesRS),
               'hhiRS': np.mean(herfRS),
               'nR': nR,
               'nR0': nR0,
               'nSXnC': nSXnC,
               'nS0XnC0': nS0XnC0,
               'lowCites': lowCites}
        
        for q in range(len(quantiles)):
            res[f"{qStr[q]}UC"] = pctList.iloc[0, q]
        for q in range(len(quantiles)):
            res[f"{qStr[q]}RS"] = np.mean(pctListRS.iloc[:, q])
        
        resALL = resALL.append(pd.Series([year] + list(res.values()), index=resALL.columns), ignore_index=True)
    
    lowCitesYears = resALL['year'][resALL['lowCites'] > lowCitesThreshold]
    if not lowCitesInclusion:
        resALL.loc[resALL['lowCites'] > lowCitesThreshold, resALL.columns.str.contains("RS")] = np.nan
        print(f"Resampled inequality measures for the years {', '.join(lowCitesYears)} recorded as NA due to smaller number of publication and citation count than the referenced year. Consider using a different base year or providing a lower refCiteCount. Set verbose=True to see more details.")
    else:
        print(f"Results for the years {', '.join(lowCitesYears)} may be incorrect due to smaller number of publication and citation count than the referenced year. Consider using a different base year or providing a lower refCiteCount. Set verbose=True to see more details.")
    
    resALL['availMetrics'] = ['everCited', 'gini', 'hhi'] + qStr
    
    return resALL

