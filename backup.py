import streamlit as st
import pandas as pd 
import numpy as np
import altair as alt
from itertools import cycle
import sqlite3
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode
import pandas as pd
from random import random
from typing import Dict, List
from streamlit import session_state as state
import streamlit as st
from streamlit import session_state as state




st.set_page_config(
    "LP Valuation",
    initial_sidebar_state="expanded",
    layout="wide",
)


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    return '%.2f%s' % (num, ['', 'k', 'm', 'b', 't', 'P'][magnitude])


# import sqlite3
# import pandas as pd

# con = sqlite3.connect("data.db")
# cur = con.cursor()
# cur.execute("DROP TABLE if exists variable_dist_0;")
# data = {'status': ['fail', 'bad', 'average', 'good', 'great'],
#     'cagr': [-0.1, 0.05, 0.1, 0.2, 0.35],
#     'management_fees': [200_000, 200_000, 400_000, 600_000, 800_000],
#     'roe': [0, 2, 4, 8, 15],
#     'distribution': [0.4, 0.15, 0.15, 0.2, 0.1]}
# df = pd.DataFrame(data)
# df.to_sql('variable_dist_0', con=con)
# con.close()

# con = sqlite3.connect("data.db")
# cur = con.cursor()
# cur.execute("DROP TABLE if exists variable_dist_1;")
# data = {'status': ['fail', 'bad', 'average', 'good', 'great'],
#     'cagr': [-0.1, 0.05, 0.1, 0.2, 0.35],
#     'management_fees': [200_000, 200_000, 350_000, 600_000, 800_000],
#     'roe': [0, 2, 4, 8, 15],
#     'distribution': [0.3, 0.15, 0.25, 0.2, 0.1]}
# df = pd.DataFrame(data)
# df.to_sql('variable_dist_1', con=con)
# con.close()

# con = sqlite3.connect("data.db")
# cur = con.cursor()
# cur.execute("DROP TABLE if exists variable_dist_2;")
# data = {'status': ['fail', 'bad', 'average', 'good', 'great'],
#     'cagr': [-0.1, 0.05, 0.15, 0.25, 0.4],
#     'management_fees': [250_000, 250_000, 400_000, 750_000, 1_000_000],
#     'roe': [0, 2, 5, 10, 18],
#     'distribution': [0.2, 0.15, 0.35, 0.2, 0.1]}
# df = pd.DataFrame(data)
# df.to_sql('variable_dist_2', con=con)
# con.close()

# con = sqlite3.connect("data.db")
# cur = con.cursor()
# cur.execute("DROP TABLE if exists variable_dist_3;")
# data = {'status': ['fail', 'bad', 'average', 'good', 'great'],
#     'cagr': [-0.1, 0.05, 0.15, 0.25, 0.5],
#     'management_fees': [250_000, 250_000, 500_000, 800_000, 1_100_000],
#     'roe': [0, 2, 5, 10, 20],
#     'distribution': [0.1, 0.15, 0.35, 0.25, 0.15]}
# df = pd.DataFrame(data)
# df.to_sql('variable_dist_3', con=con)
# con.close()

# con = sqlite3.connect("data.db")
# cur = con.cursor()
# cur.execute("DROP TABLE if exists variable_dist_4;")
# data = {'status': ['fail', 'bad', 'average', 'good', 'great'],
#     'cagr': [-0.1, 0.05, 0.15, 0.25, 0.5],
#     'management_fees': [250_000, 250_000, 600_000, 800_000, 1_100_000],
#     'roe': [0, 2, 5, 10, 20],
#     'distribution': [0.05, 0.18, 0.37, 0.25, 0.15]}
# df = pd.DataFrame(data)
# df.to_sql('variable_dist_4', con=con)
# con.close()


def reset_vars(): 
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    cur.execute("DROP TABLE if exists variable_dist;")
    data = {'status': ['fail', 'bad', 'average', 'good', 'great'],
        'cagr': [-0.1, 0.05, 0.15, 0.25, 0.5],
        'management_fees': [0, 300_000, 500_000, 750_000, 1_000_000],
        'roe': [0, 2, 5, 10, 20],
        'distribution': [0.2, 0.15, 0.35, 0.2, 0.1]}
    df = pd.DataFrame(data)
    df.to_sql('variable_dist', con=con)
    con.close()


def save(df):
    print("Saving.........")
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    cur.execute("DROP TABLE if exists variable_dist;")
    df.to_sql('variable_dist', con=con)
    con.close()


def get_dist_vars(_type):
    cols = ['status', 'cagr', 'management_fees', 'roe', 'distribution']
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    var_table_suffix = {'60%': 0, '70%': 1, '80%': 2, '90%': 3, '95%': 4}[_type]
    df = pd.DataFrame(cur.execute(f"SELECT * FROM variable_dist_{var_table_suffix}"))
    df = df.iloc[:, 1:]
    df.columns = cols
    cur.close()
    con.close()
    return df 


st.sidebar.subheader("Global Variables")
VALUATION_PERIOD: int = st.sidebar.number_input("Valuation Period", min_value=2, max_value=8, value=5)
INVESTMENT_EXPENSES_PER_DEAL: float = 250_000 #st.sidebar.number_input("Expense per Deal", min_value=100_000, max_value=1_000_000, value=500_000)
OPERATIONAL_EXPENSES: float = 1_300_000 # p.a.
OPEX_INFLATION: float = 0.05 # p.a.
DEAL_FREQ: int = st.sidebar.number_input("Deal Frequency p.a.", min_value=2, max_value=10, value=5)
INVESTMENT_SIZE: float = st.sidebar.number_input("Investment Size per Deal", min_value=1_000_000, max_value=7_500_000, value=3_500_000)
DISCOUNT_RATE: float = 0.22 #st.sidebar.number_input("Discount Rate", min_value=0.15, max_value=0.45, value=0.22)
MIN_FEE: int = 250_000 # st.sidebar.number_input("Min Management Fee", min_value=200_000, max_value=300_000, value=300_000)
MAX_FEE: int = 15_000_000 # st.sidebar.number_input("Max Management Fee", min_value=10_000_000, max_value=20_000_000, value=15_000_000)
# simulation_count = st.sidebar.number_input("Simulations", min_value=5, max_value=100_000, value=5)
with st.sidebar:
    with st.expander("Assumptions"):
        simulation_count = st.number_input("Simulations", min_value=50, max_value=100_000, value=50)
        st.caption(f"Risk Discount Rate:   {DISCOUNT_RATE*100}%")
        st.caption(f"Variable Expenses:    {human_format(INVESTMENT_EXPENSES_PER_DEAL)} per Deal")
        st.caption(f"Opex (Fixed Expenses):    {human_format(OPERATIONAL_EXPENSES)} per Annum")
        st.caption(f"Opex inflation Per Annum:    {OPEX_INFLATION*100}%")

# if 'test' not in st.session_state:
#     st.session_state['test'] = 1
# if VALUATION_PERIOD:
#     st.session_state['test'] += 1
# print(f"THE SESSION STATE IS {st.session_state}")

def get_random_key(df) -> str:
    rand: float = random()
    cum_dist: List[float] = []
    cum_value: float = 0.0
    keys = []
    dist: Dict = dict() 
    for x in df[['status', 'distribution']].to_dict('records'):
        dist.update({x['status']: x['distribution']})
        keys.append(x['status'])
    for _, val in dist.items():
        cum_value += val
        cum_dist.append(cum_value)
    for x in cum_dist:
        if rand <= x:
            # print(list(CAGR_VALUES.keys())[cum_dist.index(x)])
            # print(rand)
            return keys[cum_dist.index(x)]

sim_output: Dict = {}
class Simulate:
    def __init__(self, period, inv_size, expense, deal_freq, disc_rate, min_fee, max_fee):
        self.period = period 
        self.inv_size = inv_size
        self.expense = expense
        self.deal_freq = deal_freq 
        self.disc_rate = disc_rate 
        self.min_fee = min_fee
        self.max_fee = max_fee 


    def run(self, n_sims, df, progress_bar):
        final = pd.DataFrame()
        for k in range(int(n_sims)):
            self.results = []
            successes: int = 0
            failures: int = 0
            investment_counter: int = 0
            for i in range(1, VALUATION_PERIOD+1):
                for j in range(DEAL_FREQ):
                    deal = Investment(df, i, self.period, self.disc_rate, self.inv_size, self.max_fee)
                    if deal.cagr_rand == 'fail':
                        failures += 1
                    else: 
                        successes += 1
                    terminal_management_fee = deal.terminal_fee
                    result = {'year': i, 'deal_no': investment_counter+1, 
                            'investment_size': self.inv_size, 'expenses': self.expense,
                            'initial_fee': deal.initial_fee, 'cagr': deal.cagr, 
                            'terminal_fee': terminal_management_fee, 'equity_valuation': deal.investment_value}
                    self.results.append(result)
                    investment_counter += 1
            data = pd.DataFrame(self.results)
            global sim_output
            sim_output[f"Simulation {k}"] = data
            tmp = pd.DataFrame([{'sim_no': k, 'successes': successes, 'failures': failures,
                        'total_invested': data['investment_size'].sum(),
                        'total_expenses': data['expenses'].sum(),
                        'total_terminal_fee': data['terminal_fee'].sum(),
                        'total_equity_valuation': data['equity_valuation'].sum(),
                        'man_fee_per_investment':  data['terminal_fee'].sum()/successes,
                        'premium_inc_per_investment':  data['terminal_fee'].sum()/0.05/successes}])
            if k == 0:      
                final = tmp
            else: 
                final = pd.concat([final, tmp])
            progress_bar.progress((k+1)/n_sims)
        progress_bar.empty()
        return final



class Investment:
    def __init__(self, distribution_table: pd.DataFrame, investment_year: int, 
                period: int, discount_rate: float, inv_size: float, max_fee: float):
        self.year = investment_year
        self.cagr_rand, self.fee_rand = get_random_key(distribution_table), get_random_key(distribution_table)
        self.investment_rand = self.cagr_rand # ROI is assumed to be driven by the CAGR
        self.cagr = distribution_table.query("status == @self.cagr_rand")['cagr'].values[0] # CAGR_VALUES[self.cagr_rand]
        self.initial_fee = distribution_table.query("status == @self.fee_rand")["management_fees"].values[0]
        self.roi = distribution_table.query("status == @self.investment_rand")["roe"].values[0]
        self._terminal_fee: float = 0.0
        self._investment_value: float = 0.0
        self.period = period 
        self.discount_rate = discount_rate
        self.inv_size = inv_size
        self.max_fee = max_fee 

    def calc_terminal_fee(self):
        if self.cagr_rand == 'fail':
            self._terminal_fee = 0
        else:
            self._terminal_fee = min(self.initial_fee * pow((1+self.cagr), self.period-self.year+1), self.max_fee)


    def calc_investment_value(self):
        self._investment_value = self.inv_size * self.roi * pow(1+self.discount_rate, -(self.year-1))

    @property 
    def terminal_fee(self):
        if self._terminal_fee == 0:
            self.calc_terminal_fee()
        return self._terminal_fee

    @property
    def investment_value(self):
        self.calc_investment_value()
        return self._investment_value


def build_distribution_table():
    st.header("Launchpad Valuation Model")
    # st.subheader("Simulation Parameters")
    # st.caption("""The values in the table below can be altered by the user. Please ensure that the 'Distribution' column sums to 1.
    #             For scenario analysis change the 'CAGR', 'Management_Fees' or 'Distribution' columns. Increasing the 'Fail' distribution
    #             value to 0.4 would be a good way to conduct a stress test.""")
    # option = st.selectbox(
    #  'Select Scenario Distribution',
    #  ('Below Benchmark', 'Benchmark', 'Above Benchmark'))
    option = st.select_slider(
     'Incubation Success Ratio',
     options=['60%', '70%', '80%', '90%', '95%'], value='70%')
    df_main = get_dist_vars(option)
    return df_main


# Javascript value formatter function
def get_js(field):
    js_formatter = '''
            function(params){
                return params.data.field_name.toLocaleString(undefined, { 
            minimumFractionDigits: 0, 
            maximumFractionDigits: 0
            });
            }
            '''
    js_formatter = js_formatter.replace("field_name", field)
    return JsCode(js_formatter)



# @st.experimental_memo
def build_simulation_result_tables(df: pd.DataFrame, _progress_bar):
    x = Simulate(VALUATION_PERIOD, INVESTMENT_SIZE, INVESTMENT_EXPENSES_PER_DEAL, DEAL_FREQ, DISCOUNT_RATE, MIN_FEE, MAX_FEE).run(simulation_count, df, _progress_bar)
    return x


def build_single_sim_result_table():
    gb_single_sim = GridOptionsBuilder.from_dataframe(sim_output[list(sim_output.keys())[0]])
    gb_single_sim.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=False)
    gb_single_sim.configure_grid_options(domLayout='normal', )
    gb_single_sim.configure_column("year", header_name="Investment Year",  maxWidth = 100)
    gb_single_sim.configure_column("deal_no", header_name="Deal Number",  maxWidth = 130)
    gb_single_sim.configure_column("investment_size", header_name="Investment Made",  minWidth = 130, valueFormatter=get_js("investment_size"))
    gb_single_sim.configure_column("expenses", header_name="Expense", valueFormatter=get_js("expenses"), maxWidth=140)
    gb_single_sim.configure_column("initial_fee", header_name="Year 1 Management Fee", valueFormatter=get_js("initial_fee"))
    gb_single_sim.configure_column("cagr", header_name="CAGR", maxWidth=120)
    gb_single_sim.configure_column("terminal_fee", header_name=f"Management Fee Income Year {VALUATION_PERIOD}", minWidth=250, valueFormatter=get_js("terminal_fee"))
    gb_single_sim.configure_column("equity_valuation", header_name=f"Value of Equity in Year {VALUATION_PERIOD}", valueFormatter=get_js("equity_valuation"))
    gridOptions_single_sim = gb_single_sim.build()

    with st.expander("Single Simulation Output"):
        option = st.selectbox(
        'Select Simulation to View',
        tuple(sim_output.keys()))
        sim_table = AgGrid(sim_output[list(sim_output.keys())[0]], 
                            gridOptions = gridOptions_single_sim,
                            fit_columns_on_grid_load=True,
                            allow_unsafe_jscode=True, #Set it to True to allow jsfunction to be injected
                            enable_enterprise_modules=True,
                            theme='dark')



def build_distribution_charts(x):
    c = x['total_terminal_fee'].hist(bins=30, backend='plotly')
    c.update_layout(showlegend=False)
    c2 = x['total_equity_valuation'].hist(bins=30, backend='plotly')
    c2.update_layout(showlegend=False)
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        with st.expander("Management Fee Distribution", expanded=True):
            st.plotly_chart(c, use_container_width=True)

    with dist_col2:
        with st.expander("Equity Value Distribution", expanded=True):
            st.plotly_chart(c2, use_container_width=True)




def build_sim_metrics(x):
    st.subheader("Valuation Summary")
    total_invested = x['total_invested'].mean()
    PE_ratio = 10.5 
    valuation_man_fees_total = x['total_terminal_fee'].mean() * PE_ratio
    valuation_equity_total = x['total_equity_valuation'].mean()
    variable_expenses = INVESTMENT_EXPENSES_PER_DEAL * VALUATION_PERIOD * DEAL_FREQ
    opex_total = sum([OPERATIONAL_EXPENSES*pow((1 + OPEX_INFLATION), i) for i in range(VALUATION_PERIOD)])
    net_invested = valuation_man_fees_total + valuation_equity_total - (x['total_invested'].mean() + variable_expenses + opex_total)
    x["total_valuation_value"] = (x['total_terminal_fee'] * PE_ratio) + x['total_equity_valuation'] - (x['total_invested'] + variable_expenses + opex_total)

    import plotly.graph_objects as go

    pie_labels = ['Guardrisk Earnings <br> Contribution', 'Launchpad Equity <br> Contribution']
    pie_values = [valuation_man_fees_total, valuation_equity_total]
    layout = go.Layout(
        margin=go.layout.Margin(
            l=0, #left margin
                r=200, #right margin
                b=50, #bottom margin
                t=0  #top margin
            ),
        )
    fig = go.Figure(data=[go.Pie(labels=pie_labels, values=pie_values, textinfo='label+percent', hole=.3)], layout=layout)
    fig.update_layout(showlegend=False)

    with st.expander("Headline Net Valuation Result", expanded=True):
        st.metric("Launchpad + Guardrisk Value", human_format(net_invested))
        col1, _, col3 = st.columns(3)
        with col1:
            fig
        with col3:
            st.metric("Earnings Contribution to Total Valuation", human_format(valuation_man_fees_total))
            st.metric("Total Portfolio Equity Valuation", human_format(valuation_equity_total))
            st.metric("Total Invested", '(' + human_format(total_invested) + ')')
            st.metric("Operational Expenses", '(' + human_format(variable_expenses) + ')')
            st.metric("Variable Expenses (Per Deal)", '(' + human_format(opex_total) + ')')
        dist_graph_total_value = x['total_valuation_value'].hist(bins=30, backend='plotly')
        dist_graph_total_value.update_layout(showlegend=False)
        dist_graph_total_value.update_layout(margin=go.layout.Margin(
            l=0, #left margin
                r=100, #right margin
                b=0, #bottom margin
                t=0  #top margin
            ),)
        st.subheader("Distribution of Outcomes")
        dist_col1, dist_col2 = st.columns([2,4])
        with dist_col2:
            st.plotly_chart(dist_graph_total_value, use_container_width=True)

        with dist_col1:
            quantiles = ["{:,}".format(int(x['total_valuation_value'].quantile(q=i))) for i in np.arange(0.05,1,0.05)]
            quantile_names = [str(round(i*100,0)) + '%' for i in np.arange(0.05,1,0.05)]
            percentile_df = pd.DataFrame({"Percentile": quantile_names, "Net Valuation Total": quantiles})
            percentile_df.style.set_properties(subset=['Percentile'], **{'text-align': 'right'})
            st.write(percentile_df)

    with st.expander("Result Metrics", expanded=True):
        col_summary5, col_summary6, col_summary7, col_summary8 = st.columns(4)
        with col_summary5:
            st.metric("Average Management Fee per cell p.a.", human_format(x['man_fee_per_investment'].mean()))
        with col_summary6:
            st.metric("Average Premium per cell p.a.", human_format(x['premium_inc_per_investment'].mean()))
        with col_summary7:
            st.metric("Total Successes", f"{x['successes'].mean()}")
        with col_summary8:
            st.metric("Total Failures", f"{x['failures'].mean()}")



def main():
    ...


if __name__ == "__main__":
    df = build_distribution_table()
    _progress_bar = st.progress(0)
    sim_result_data = build_simulation_result_tables(df, _progress_bar)
    build_sim_metrics(sim_result_data)
    build_distribution_charts(sim_result_data)
    
    
