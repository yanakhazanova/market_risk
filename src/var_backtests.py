import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, chi2, binom
import warnings
warnings.filterwarnings('ignore')


def dq_bt(df, alpha, hit_lags=4, forecast_lags=1):
    try:
        hits = df['Violation']  # Assuming 'hit_series()' get these values
        p, q, n = hit_lags, forecast_lags, hits.size
        pq = max(p, q - 1)
        y = hits[pq:] - alpha  # Dependent variable
        x = np.zeros((n - pq, 1 + p + q))
        x[:, 0] = 1  # Constant

        for i in range(p):  # Lagged hits
            x[:, 1 + i] = hits[pq-(i+1):-(i+1)]

        forecast = df['VaR']  # Assuming 'forecast' is equal to 'VaR'

        for j in range(q):  # Actual + lagged VaR forecast
            if j > 0:
                x[:, 1 + p + j] = forecast[pq-j:-j]
            else:
                x[:, 1 + p + j] = forecast[pq:]

        beta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
        lr_dq = np.dot(beta, np.dot(np.dot(x.T, x), beta)) / (alpha * (1 - alpha))
        p_dq = 1 - stats.chi2.cdf(lr_dq, 1 + p + q)
    except Exception as e:
        print("Exception occurred: ", e)  # print exception for debugging
        lr_dq, p_dq = np.nan, np.nan
    result = pd.Series([lr_dq, p_dq], index=["Statistic", "p-value"], name="DQ")
    # Check hypothesis
    if p_dq < alpha:
        print("Null Hypothesis H0 is rejected")
    else:
        print("Null Hypothesis H0 is not rejected")
    return result


def lr_bt(df, alpha):
    """Likelihood ratio framework of Christoffersen (1998)"""
    hits, tr = df['Violation'].values[1:], df['Violation'].values[:-1] - df['Violation'].values[1:]

    n01, n10 = np.sum(tr == 1), np.sum(tr == -1)
    n11, n00 = np.sum(hits[tr == 0] == 1), np.sum(hits[tr == 0] == 0)

    n0, n1 = n01 + n00, n10 + n11
    p01, p11 = n01 / n0, n11 / n1
    p = n1 / (n0 + n1)

    if n1 > 0:
        log_1_alpha, log_alpha, log_1_p, log_p = np.log(1 - alpha), np.log(alpha), np.log(1 - p), np.log(p)
        uc_h0, uc_h1 = n0 * log_1_alpha + n1 * log_alpha, n0 * log_1_p + n1 * log_p
        uc = -2 * (uc_h0 - uc_h1)
        print("For Unconditional Coverage, Null Hypothesis H0 is", "rejected" if 1 - stats.chi2.cdf(uc, 1) < alpha else "not rejected")

        ind_h0 = (n00 + n01) * log_1_p + (n01 + n11) * log_p
        ind_h1 = n0 * np.log(1 - p01) + (n01 * np.log(p01) + n10 * np.log(1 - p11)) + n11 * np.log(p11) if p11 > 0 else 0
        ind = -2 * (ind_h0 - ind_h1)
        print("For Independence, Null Hypothesis H0 is", "rejected" if 1 - stats.chi2.cdf(ind, 1) < alpha else "not rejected")

        cc = uc + ind
        print("For Conditional Coverage, Null Hypothesis H0 is", "rejected" if 1 - stats.chi2.cdf(cc, 2) < alpha else "not rejected")

        df = pd.concat([pd.Series([uc, ind, cc]), pd.Series([1 - stats.chi2.cdf(uc, 1), 1 - stats.chi2.cdf(ind, 1),
                                                              1 - stats.chi2.cdf(cc, 2)])], axis=1)
    else:
        df = pd.DataFrame(np.zeros((3, 2))).replace(0, np.nan)

    df.columns, df.index = ["Statistic", "p-value"], ["Unconditional", "Independence", "Conditional"]
    return df


def haas_bt(df, var_column='VaR', pnl_column='PnL'):
    """ Haas(2006) """
    # get necessary series
    var = df[var_column]
    pnl = df[pnl_column]

    # count breaches
    breaches = (pnl < var).sum()

    # total observations
    obs = df.shape[0]

    # Test statistic: Haas' likelihood ratio
    LRuc = -2 * ((1 - var).apply(np.log).sum() - obs * np.log(1 - breaches / obs))

    # Asymptotic p-value
    p_value = 1 - stats.chi2.cdf(LRuc, 1)

    print("For Unconditional Coverage, Null Hypothesis H0 is", "rejected" if p_value < 0.05 else "not rejected")

    df = pd.Series([LRuc, p_value], index=["Statistic", "p-value"], name="Haas Test")

    return df


def dumitrescu_backtest(df, var_column='VaR', pnl_column='PnL', confidence_level=0.05):
    losses = df[pnl_column].values
    var_estimates = df[var_column].values
    n = len(losses)

    # Count exceedances (breaches)
    exceedances = losses > var_estimates

    # Compute the conditional coverage probability
    ccp = np.mean(exceedances)

    # Expected ccp
    expected_ccp = 1 - norm.cdf(norm.ppf(1-confidence_level))

    # Test statistic and p-value
    test_statistic = 2 * n * (ccp - expected_ccp)
    p_value = 1 - chi2.cdf(test_statistic, 1)

    # Null hypothesis test
    hypothesis_result = "Rejected" if p_value < confidence_level else "Failed to reject"

    print("Test Statistic:", test_statistic)
    print("Test p-value:", p_value)
    print("Null Hypothesis H0 is", hypothesis_result)

    return test_statistic, p_value


def candelon_backtest(df, var_column='VaR', pnl_column='PnL', confidence_level=0.05):
    losses = df[pnl_column].values
    var_estimates = df[var_column].values
    n = len(losses)

    # Count violations (breaches)
    violations = losses > var_estimates
    violation_count = violations.sum()

    # Calculate proportions
    violation_proportion = violation_count / n
    expected_proportion = 1 - confidence_level

    # Test statistic and p-value
    test_statistic = 2 * n * (violation_proportion - expected_proportion)
    p_value = 1 - chi2.cdf(test_statistic, 1)

    # Hypothesis result
    hypothesis_result = "Rejected" if p_value < confidence_level else "Failed to reject"

    print("Test Statistic:", test_statistic)
    print("Test p-value:", p_value)
    print("Null Hypothesis H0:", hypothesis_result)

    return test_statistic, p_value

def colletaz_backtest(df, var_column='VaR', pnl_column='PnL', confidence_level=0.05):
    losses = df[pnl_column].values
    var_estimates = df[var_column].values
    n = len(losses)

    # Calculate violations
    violations = losses > var_estimates
    durations = []
    violation_count = 0

    for violation in violations:
        if violation:
            violation_count += 1
        else:
            if violation_count > 0:
                durations.append(violation_count)
                violation_count = 0

    # Expected number of violations
    expected_violations = binom.ppf(1 - confidence_level, n, 1 / (1 + np.mean(durations)))

    # Test statistic and p-value
    test_statistic = np.square(violation_count - expected_violations) / expected_violations
    p_value = 1 - chi2.cdf(test_statistic, len(durations) - 1)

    # Hypothesis result
    hypothesis_result = "Rejected" if p_value < confidence_level else "Failed to reject"

    print("Test Statistic:", test_statistic)
    print("Test p-value:", p_value)
    print("Null Hypothesis H0:", hypothesis_result)

    return test_statistic, p_value


def pelletier_wei_backtest(df, var_column='VaR', pnl_column='PnL', violation_column='Violation', confidence_level=0.05):
    # Calculate the number of violations
    V = sum(df[violation_column])

    # Calculate the expected number of violations
    E_V = sum(df[var_column] < -df[pnl_column])

    # Calculate the variance of the number of violations
    n = len(df)
    Var_V = E_V * (1 - E_V / n) * (n - E_V) / (n - 1)

    # Calculate the test statistic
    Z = (V - E_V) / np.sqrt(Var_V)

    # Calculate the p-value
    p_value = 2 * (1 - norm.cdf(abs(Z)))

    # Hypothesis result
    hypothesis_result = "Rejected" if p_value < confidence_level else "Failed to reject"

    print("Test Statistic:", Z)
    print("Test p-value:", p_value)
    print("Null Hypothesis H0:", hypothesis_result)

    return {"Test Statistic": Z, "p-value": p_value, "Null Hypothesis H0": hypothesis_result}
