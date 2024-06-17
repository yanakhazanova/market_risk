import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import pandas_datareader as pdr
import yfinance as yf
import plotly.graph_objects as go
import datetime as dt
import seaborn as sns
import warnings
import typing
import rich
import investpy
import arch
from tqdm import tqdm
import scipy.stats as ss
from arch.univariate import GARCH, ConstantMean, SkewStudent
from sklearn.ensemble import GradientBoostingRegressor
from arch import arch_model
from functools import partial
from statsmodels.tsa.seasonal import seasonal_decompose
from rich import print
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import PhillipsPerron
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
import warnings
import xgboost as xgb
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

RANDOM_STATE = 228
np.random.seed(RANDOM_STATE)

def calculate_VaR_hs(ret, alpha=0.05):
    return ret.quantile(alpha)

def sim_hs(risk_factor, n_samples=10**3):
    return np.random.choice(risk_factor, n_samples)

def bern_test(p, v):
    lv = len(v)
    sv = sum(v)
    al = np.log(p)*sv + np.log(1-p)*(lv-sv)
    bl = np.log(sv/lv)*sv + np.log(1-sv/lv)*(lv-sv)
    return (-2*(al-bl))

def ind_test(V):
    J = np.full([len(V),4], 0)
    V = V.values
    for i in range(1,len(V)-1):
        J[i,0] = (V[i-1] == 0) & (V[i] == 0)
        J[i,1] = (V[i-1] == 0) & (V[i] == 1)
        J[i,2] = (V[i-1] == 1) & (V[i] == 0)
        J[i,3] = (V[i-1] == 1) & (V[i] == 1)
    V_00 = sum(J[:,0])
    V_01 = sum(J[:,1])
    V_10 = sum(J[:,2])
    V_11 = sum(J[:,3])
    p_00=V_00/(V_00+V_01)
    p_01=V_01/(V_00+V_01)
    p_10=V_10/(V_10+V_11)
    p_11=V_11/(V_10+V_11)
    hat_p = (V_01+V_11)/(V_00+V_01+V_10+V_11)
    al = np.log(1-hat_p)*(V_00+V_10) + np.log(hat_p)*(V_01+V_11)
    bl = np.log(p_00)*V_00 + np.log(p_01)*V_01 + np.log(p_10)*V_10 + np.log(p_11)*V_11
    return (-2*(al-bl))

def create_log_diff_data(data, horizon=1):
    df_log = pd.DataFrame()
    for col in data.columns:
        df_log[col] = data[col].iloc[horizon:].values / data[col].iloc[:-horizon].values - 1
    df_log.dropna(inplace=True)
    df_log.set_index(data.index[horizon:], inplace=True)
    return df_log

def get_number_of_assets(df_portfolio, horizon=1):
    investment = np.array([10e6] * 5 + [1e6] * 10 + [10e6] * 2)
    prices = df_portfolio.iloc[0]
    quantity = np.floor(investment / prices).astype(int)
    proportion = investment / investment.sum()
    number_of_assets = quantity
    all_price = prices

    for t in range(df_portfolio.shape[0] - 1): # поменял тут -1 на -horizon
        prices = df_portfolio.iloc[t]
        weights = (quantity * prices) / np.sum(quantity * prices)
        delta_weights = proportion / weights
        quantity = np.floor(quantity * delta_weights).astype(int)
        number_of_assets = np.vstack((number_of_assets, quantity))
        all_price = np.vstack((all_price, prices))
    number_of_assets = pd.DataFrame(number_of_assets[horizon:], columns=df_portfolio.columns, index=df_portfolio.index[horizon: ])
    return number_of_assets

def make_riskfactors_simulations(risk_factors, n_samples=10**3):
    simulated_riskfactors = pd.DataFrame()
    is_CIR = 0
    for column in risk_factors.columns:
        if sm.tsa.stattools.adfuller(risk_factors[column])[1] > 0.05:
            is_CIR_ = 1
    if is_CIR == 0:
        for column in risk_factors.columns:
            simulated_riskfactors[column] = sim_hs(risk_factors[column], n_samples)
    elif is_CIR == 1:
        simulated_riskfactors = CIRSimulation(df = risk_factors, period_length = 2000, num_simulations = n_samples)
    return simulated_riskfactors


class VaRModel:
    def __init__ (self, df_portfolio, df_risk_factors, assets, horizon=1, history_length=252, num_sims=10000):
        self.df_portfolio = df_portfolio[assets]
        self.risk_factors = df_risk_factors
        self.horizon = horizon
        self.history_length = history_length
        self.num_sims = num_sims
        self.df_log_diff_portfolio = create_log_diff_data(self.df_portfolio, self.horizon)
        self.number_of_assets = get_number_of_assets(df_portfolio, self.horizon)[assets]
        self.w = (self.number_of_assets * self.df_portfolio[self.df_log_diff_portfolio.columns].iloc[
                                          self.horizon:]).div(
            pd.DataFrame((self.number_of_assets * self.df_portfolio[self.df_log_diff_portfolio.columns].iloc[
                                                   self.horizon:]).sum(axis=1)).values
        )

    def get_VaR_and_ES (self):
        VaR = np.full(self.df_log_diff_portfolio.shape, np.nan)
        VaR_for_ES = np.full(self.df_log_diff_portfolio.shape, np.nan)
        ES = np.full(self.df_log_diff_portfolio.shape, np.nan)
        self.dict_gb = {}
        for column in self.df_log_diff_portfolio.columns:
            GB = xgb.XGBRegressor(seed=RANDOM_STATE, objective='reg:squarederror')
            GB.fit(self.risk_factors[self.horizon:], self.df_log_diff_portfolio[column])
            self.dict_gb[column] = GB
        for i in tqdm(range(self.history_length, len(self.df_log_diff_portfolio) - 1)):
            history_risk_factors = self.risk_factors.iloc[i - self.history_length: i + 1, :]
            sims = make_riskfactors_simulations(history_risk_factors, self.num_sims)
            for j, column in enumerate (self.df_log_diff_portfolio.columns):
                pred = self.dict_gb[column].predict(sims)
                VaR [i + 1, j] = np.quantile (pred, 0.01)
                VaR_for_ES [i + 1, j] = np.quantile (pred, 0.025)
                history = self.df_log_diff_portfolio [column] [i - self.history_length: i + 1]
                ES [i + 1, j] = history [history <= VaR_for_ES [i + 1, j]].mean ()

        VaR_per_asset = pd.DataFrame (data=VaR, index=self.df_log_diff_portfolio.index,
                                      columns=self.df_log_diff_portfolio.columns)
        ES_per_asset = pd.DataFrame (data=ES, index=self.df_log_diff_portfolio.index,
                                     columns=self.df_log_diff_portfolio.columns)

        VaR = pd.Series (data=(VaR_per_asset * self.w).sum (axis=1), index=self.df_log_diff_portfolio.index)
        ES = pd.Series (data=(ES_per_asset * self.w).sum (axis=1), index=self.df_log_diff_portfolio.index)
        VaR[VaR == 0] = np.nan
        ES[ES == 0] = np.nan
        self.VaR = VaR
        self.ES = ES
        self.VaR_per_asset = VaR_per_asset
        self.ES_per_asset = ES_per_asset
        return VaR, ES, VaR_per_asset, ES_per_asset

    def plot_VaR_ES(self, plot_title=''):
        plt.figure(figsize=(16, 8))
        lr = pd.DataFrame((self.df_log_diff_portfolio.to_numpy() * self.w.to_numpy()).sum(axis=1))
        var = pd.DataFrame(
            (self.VaR.to_numpy().reshape(-1, 1) * self.w.to_numpy()).sum(axis=1)[self.history_length + 1:])
        es = pd.DataFrame(
            (self.ES.to_numpy().reshape(-1, 1) * self.w.to_numpy()).sum (axis=1)[self.history_length + 1:])
        var = pd.DataFrame(np.hstack([np.full(len(lr) - len(var), np.nan), var.iloc[:, 0].to_numpy()]))
        es = pd.DataFrame(np.hstack([np.full(len(lr) - len(es), np.nan), es.iloc[:, 0].to_numpy()]))
        plot_df = pd.merge(lr, var, how='outer', left_index=True, right_index=True)
        plot_df = pd.merge(plot_df, es, how='outer', left_index=True, right_index=True)
        plot_df.columns = ['Daily Weighted Log Returns', 'VaR', 'ES']
        fig = px.line(plot_df)
        fig.update_layout(
            title=plot_title
        )
        fig.show()
        plt.show()

    def calc_fair_price(self, column, mode = 'mean'):
        preds = np.full(self.df_log_diff_portfolio.shape, np.nan)
        pred_sims = np.full((self.df_log_diff_portfolio.shape[0] - self.history_length - 1, self.num_sims), np.nan)
        for i in tqdm(range(self.history_length, self.df_log_diff_portfolio.shape[0] - 1)):
            history_risk_factors = self.risk_factors.iloc[i - self.history_length:i + 1, :]
            sims = make_riskfactors_simulations(history_risk_factors, self.num_sims)
            pred = self.dict_gb[column].predict(sims).reshape(-1)
            if mode == 'mean':
                preds[i + 1] = np.mean(pred)
            elif mode == 'median':
                preds[i + 1] = np.median(pred)
            else:
                preds[i + 1] = np.quantile(pred, mode)
        preds = preds[:, 0]
        preds = pd.Series (data=preds, index=self.df_log_diff_portfolio.index)
        preds[preds == 0] = np.nan
        fair_price = pd.DataFrame(
        np.full(self.VaR_per_asset.shape, np.nan),
        columns = self.VaR_per_asset.columns,
        index = self.VaR_per_asset.index
        )
        ind = np.where(fair_price.columns == column)[0][0]
        for row in range(self.history_length+1, self.VaR_per_asset.shape[0]):
            fair_price.iloc[row, ind] = np.exp(preds.iloc[row])+ self.df_portfolio[column].iloc[row-self.horizon]
        return fair_price.iloc[:, ind]
    
    def get_all_fair_prices(self, mode = 'mean'):
        fair_price = pd.DataFrame(
            np.full(self.VaR_per_asset.shape, np.nan),
            columns = self.VaR_per_asset.columns,
            index = self.VaR_per_asset.index
            )
        for column in self.df_log_diff_portfolio.columns:
            ind = np.where(fair_price.columns == column)[0][0]
            fair_price.iloc[:, ind] = self.calc_fair_price(column)
        return fair_price
    
    def backtest (self, alpha=0.01, significance=0.95):
        lr = pd.DataFrame ((self.df_log_diff_portfolio.to_numpy () * self.w.to_numpy ()).sum (axis=1))
        var_curve = self.VaR
        var_curve.index = lr.index
        idx = var_curve.notna ()
        violations = lr [idx] < pd.DataFrame (var_curve [idx])
        violations = violations.astype (int)
        coverage = bern_test (p=alpha, v=violations) < ss.chi2.ppf (significance, 1)
        independence = ind_test (violations) < ss.chi2.ppf (significance, 1)
        print ('Number of violations: ', violations.sum ().item ())
        print ('Target share of violations: {:.2f}%'.format (100 * alpha))
        print ('Observed share of violations: {:.2f}%'.format (100 * violations.mean ().item ()))
        print ()
        if coverage:
            print ('Test for coverage: Passed')
        else:
            print ("Test for coverage: Not Passed")
        print ()
        if independence:
            print ("Test for independence: Passed")
        else:
            print ("Test for independence: Not Passed")
     
class CIRParams:
    """CIR process params class."""

    a: float  # mean reversion parameter
    b: float  # asymptotic mean
    c: float  # Brownian motion scale factor (standard deviation)

    def __init__(self, sigma_t: np.ndarray):
        # define regression specification
        sigma_sqrt: np.ndarray = np.sqrt(sigma_t[:-1])
        y: np.ndarray = np.diff(sigma_t) / sigma_sqrt
        x1, x2 = 1.0 / sigma_sqrt, sigma_sqrt
        X = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1)], axis=1)

        # regression model
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        # residuals and their standard deviation
        y_hat = reg.predict(X)
        # regression coefficients
        ab = reg.coef_[0]

        self.a = float(-(reg.coef_[1]))
        self.b = float(ab / self.a)
        self.c = float(np.std(y - y_hat))

        assert 2 * self.a * self.b >= self.c ** 2, 'failed check: 2ab < c^2'


class CIRSimulation:
    def __init__(
            self, df: pd.DataFrame, period_length: int, num_simulations: int,
    ):
        self.df = df
        self.df['ln_rate'] = np.log(self.df.iloc[:, 0])
        self.period_length = period_length
        self.num_simulations = num_simulations
        self.scaler = StandardScaler()

    def get_correlated_simulation(self) -> np.ndarray:
        def make_zeros():
            return np.zeros((self.period_length, self.num_simulations, 3))

        sim_arr = make_zeros()
        dW_arr = make_zeros()
        ret_arr = make_zeros()

        first_comp = CIRParams(self.df.iloc[:, 0].values)
        second_comp = CIRParams(self.df.iloc[:, 1].values)
        third_comp = CIRParams(self.df.iloc[:, 2].values)
        forth_comp = CIRParams(self.df.iloc[:, 3].values)
        fifth_comp = CIRParams(self.df.iloc[:, 4].values)

        for i in range(self.num_simulations):
            sim_arr[:, i, 0], dW_arr[:, i, 0] = self._generate_cir_process(
                cir_params=first_comp, sigma_0=self.df.iloc[-1, 0],
            )
            sim_arr[:, i, 1], dW_arr[:, i, 1] = self._generate_cir_process(
                cir_params=second_comp, sigma_0=self.df.iloc[-1, 1],
            )
            sim_arr[:, i, 2], dW_arr[:, i, 2] = self._generate_rate_process(
                cir_params=third_comp, sigma_0=self.df.iloc[-1, 2],
            )
            
            sim_arr[:, i, 3], dW_arr[:, i, 2] = self._generate_rate_process(
                cir_params=forth_comp, sigma_0=self.df.iloc[-1, 3],
            )
            
            sim_arr[:, i, 4], dW_arr[:, i, 2] = self._generate_rate_process(
                cir_params=fifth_comp, sigma_0=self.df.iloc[-1, 4],
            )

            cor_matrix = np.linalg.cholesky(
                pd.DataFrame(dW_arr[:, i, :]).corr(),
            )
            scaled_arr = self.scaler.fit_transform(sim_arr[:, i, :])
            ret_arr[:, i, :] = scaled_arr @ cor_matrix
            ret_arr[:, i, :] = self.scaler.inverse_transform(ret_arr[:, i, :])

        return ret_arr

    def _generate_cir_process(
            self, cir_params: CIRParams, sigma_0: float,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:

        dW: np.ndarray = np.random.normal(0.0, 1.0, self.period_length)
        sigma_t: typing.List[float] = [sigma_0]

        for t in range(1, self.period_length):
            dsigma_t = (
                cir_params.a * (cir_params.b - sigma_t[t - 1])
                + cir_params.c * np.sqrt(sigma_t[t - 1]) * dW[t]
            )
            sigma_t.append(sigma_t[t - 1] + dsigma_t)

        return np.asarray(sigma_t), dW

    def _generate_rate_process(
            self, cir_params: CIRParams, sigma_0: float,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:

        dW: np.ndarray = np.random.normal(0.0, 1.0, self.period_length)
        sigma_t: typing.List[float] = [sigma_0]

        for t in range(1, self.period_length):
            dsigma_t = (
                cir_params.a * (cir_params.b - sigma_t[t - 1])
                + cir_params.c * dW[t]
            )
            sigma_t.append(sigma_t[t - 1] + dsigma_t)

        return np.exp(np.asarray(sigma_t)), dW