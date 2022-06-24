from soupsieve import select
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
    tmp = num
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # add more suffixes if you need them
    if tmp < 10_000_000:
        return '%.2f%s' % (num, ['', 'k', 'm', 'm', 't', 'P'][magnitude])
    else:
        if magnitude == 3:
            num = num*1000
        return '{:,}{}'.format(int(num), ['', 'k', 'm', 'm', 't', 'P'][magnitude])



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

DIST_SELECTION = {}
INTERMEDIATE: Dict = [DIST_SELECTION.update({f'{x}%': i}) for i, x in enumerate(np.arange(60,100,5))][0]



def get_dist_vars(_type):
    cols = ['isr', 'status', 'cagr', 'management_fees', 'roe', 'distribution', 'type']
    con = sqlite3.connect("data.db")
    cur = con.cursor()
    # var_table_suffix = DIST_SELECTION[_type]
    df = pd.DataFrame(cur.execute(f"SELECT * FROM sim_data WHERE type = {_type}"))
    # df = df.iloc[:, 1:]
    df.columns = cols
    cur.close()
    con.close()
    return df 


st.sidebar.subheader("Global Variable")
VALUATION_PERIOD: int = st.sidebar.number_input("Valuation Period", min_value=2, max_value=8, value=5)
OPEX_INFLATION: float = 0.05 # p.a.
DEAL_FREQ: int = st.sidebar.number_input("Deal Volume", min_value=2, max_value=10, value=5)
INVESTMENT_SIZE: float = st.sidebar.number_input("Investment Size per Deal", min_value=1_000_000, max_value=7_500_000, value=3_500_000)
st.sidebar.caption("Investment size: {:,}".format(INVESTMENT_SIZE))
INVESTMENT_EXPENSES_PER_DEAL: float = st.sidebar.number_input("Variable Expenses per Deal", min_value=50_000, max_value=1_500_000, value=250_000)
st.sidebar.caption("Variable expenses: {:,}".format(INVESTMENT_EXPENSES_PER_DEAL))
OPERATIONAL_EXPENSES: float = st.sidebar.number_input("Operational Expenses per Annum", min_value=500_000, max_value=7_500_000, value=1_300_000)
st.sidebar.caption("Operational expenses: {:,}".format(OPERATIONAL_EXPENSES))
DISCOUNT_RATE: float = st.sidebar.number_input("Discount Rate", min_value=0.15, max_value=0.45, value=0.22)
MIN_FEE: int = 250_000 # st.sidebar.number_input("Min Management Fee", min_value=200_000, max_value=300_000, value=300_000)
MAX_FEE: int = 15_000_000 # st.sidebar.number_input("Max Management Fee", min_value=10_000_000, max_value=20_000_000, value=15_000_000)
# simulation_count = st.sidebar.number_input("Simulations", min_value=5, max_value=100_000, value=5)
with st.sidebar:
    with st.expander("Hidden Variable"):
        simulation_count = st.number_input("Simulations", min_value=50, max_value=1000, value=50)
        # st.caption(f"Risk Discount Rate:   {DISCOUNT_RATE*100}%")
        # st.caption(f"Variable Expenses:    {human_format(INVESTMENT_EXPENSES_PER_DEAL)} per Deal")
        # st.caption(f"Opex (Fixed Expenses):    {human_format(OPERATIONAL_EXPENSES)} per Annum")
        # st.caption(f"Opex inflation Per Annum:    {OPEX_INFLATION*100}%")

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


values = []
def get_dist_tables(_type):
    global values
    selected_distribution = DIST_SELECTION[_type]
    if selected_distribution == 7:
        values = [7 for _ in range(4)]
    elif selected_distribution == 6:
        values = [6,6,7,7]
    elif selected_distribution == 5:
        values = [5,6,6,7]
    else:
        values = [selected_distribution, selected_distribution + 1, selected_distribution + 1, selected_distribution + 2]
    df, df1, df2, df3 = (get_dist_vars(x) for x in values)
    return (df, df1, df2, df3)

class Simulate:
    def __init__(self, period, inv_size, expense, deal_freq, disc_rate, min_fee, max_fee):
        self.period = period 
        self.inv_size = inv_size
        self.expense = expense
        self.deal_freq = deal_freq 
        self.disc_rate = disc_rate 
        self.min_fee = min_fee
        self.max_fee = max_fee 


    def run(self, n_sims, dfs, progress_bar):
        final = pd.DataFrame()
        for k in range(int(n_sims)):
            self.results = []
            successes: int = 0
            failures: int = 0
            investment_counter: int = 0
            for i in range(1, int(VALUATION_PERIOD)+1):
                for j in range(int(DEAL_FREQ)):
                    deal = Investment(dfs, i, self.period, self.disc_rate, self.inv_size, self.max_fee)
                    if deal.cagr_rand[0] == 'fail':
                        failures += 1
                    else: 
                        successes += 1
                    terminal_management_fee = deal.terminal_fee
                    result = {'year': i, 'deal_no': investment_counter+1, 
                            'investment_size': self.inv_size, 'expenses': self.expense,
                            'initial_fee': deal.initial_fee[0], 'cagr': deal.cagr[0], 
                            'terminal_fee': terminal_management_fee[0], 
                            'terminal_fee2': terminal_management_fee[1], 
                            'terminal_fee3': terminal_management_fee[2], 
                            'terminal_fee4': terminal_management_fee[3], 
                            'equity_valuation': deal.investment_value[0],
                            'equity_valuation2': deal.investment_value[1],
                            'equity_valuation3': deal.investment_value[2],
                            'equity_valuation4': deal.investment_value[3]}
                    self.results.append(result)
                    investment_counter += 1
            data = pd.DataFrame(self.results)
            global sim_output
            sim_output[f"Simulation {k}"] = data
            tmp = pd.DataFrame([{'sim_no': k, 'successes': successes, 'failures': failures,
                        'total_invested': data['investment_size'].sum(),
                        'total_expenses': data['expenses'].sum(),
                        'total_terminal_fee0': data['terminal_fee'].sum(),
                        'total_terminal_fee1': data['terminal_fee2'].sum(),
                        'total_terminal_fee2': data['terminal_fee3'].sum(),
                        'total_terminal_fee3': data['terminal_fee4'].sum(),
                        'total_equity_valuation0': data['equity_valuation'].sum(),
                        'total_equity_valuation1': data['equity_valuation2'].sum(),
                        'total_equity_valuation2': data['equity_valuation3'].sum(),
                        'total_equity_valuation3': data['equity_valuation4'].sum(),
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
    def __init__(self, distribution_tables: pd.DataFrame, investment_year: int, 
                period: int, discount_rate: float, inv_size: float, max_fee: float):
        # self.dist_table, self.dist_table2, self.dist_table3, self.dist_table4 = distribution_tables     
        self.year = investment_year
        self.cagr_rand, self.fee_rand = [get_random_key(distribution_tables[x]) for x in range(4)], [get_random_key(distribution_tables[x]) for x in range(4)]
        self.investment_rand = self.cagr_rand # ROI is assumed to be driven by the CAGR
        # self.cagr = [distribution_tables[x].query("status == @self.cagr_rand[x]")['cagr'].values[0] for x in range(4)] # CAGR_VALUES[self.cagr_rand]
        # self.initial_fee = [distribution_tables[x].query("status == @self.fee_rand[x]")["management_fees"].values[0] for x in range(4)]
        # self.roi = [distribution_tables[x].query("status == @self.investment_rand[x]")["roe"].values[0] for x in range(4)]

        self.cagr = [distribution_tables[x][distribution_tables[x]["status"] == self.cagr_rand[x]]['cagr'].values[0] for x in range(4)] # CAGR_VALUES[self.cagr_rand]
        self.initial_fee = [distribution_tables[x][distribution_tables[x]["status"] == self.fee_rand[x]]["management_fees"].values[0] for x in range(4)]
        self.roi = [distribution_tables[x][distribution_tables[x]["status"] == self.investment_rand[x]]["roe"].values[0] for x in range(4)]
        self._terminal_fee: List[float] = [0,0,0,0]
        self._investment_value: List[float] = []
        self.period = period 
        self.discount_rate = discount_rate
        self.inv_size = inv_size
        self.max_fee = max_fee 

    def calc_terminal_fee(self):
        # if self.cagr_rand == 'fail':
        #     self._terminal_fee = [0,0,0,0]
        # else:
        self._terminal_fee = [min(self.initial_fee[x] * pow((1+self.cagr[x]), self.period-self.year+1), self.max_fee) if self.cagr_rand[x] != 'fail' else 0 for x in range(4)]
        # print(self._terminal_fee)


    def calc_investment_value(self):
        self._investment_value = [self.inv_size * self.roi[x] * pow(1+self.discount_rate, -(self.year-1)) if self.cagr_rand[x] != 'fail' else 0 for x in range(4)]

    @property 
    def terminal_fee(self):
        if self._terminal_fee == [0,0,0,0]:
            self.calc_terminal_fee()
        return self._terminal_fee

    @property
    def investment_value(self):
        self.calc_investment_value()
        return self._investment_value


def build_distribution_table():
    st.header("Launchpad Valuation Model")
    option = st.select_slider(
     'Incubation Success Ratio',
     options=['60%', '65%', '70%', '75%', '80%', '85%', '90%', '95%'], value='70%')
    # val = DIST_SELECTION[option] 
    df_main = get_dist_tables(option) #get_dist_vars(option)
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
def build_simulation_result_tables(df: tuple, _progress_bar):
    x = Simulate(VALUATION_PERIOD, INVESTMENT_SIZE, INVESTMENT_EXPENSES_PER_DEAL, DEAL_FREQ, DISCOUNT_RATE, MIN_FEE, MAX_FEE).run(simulation_count, dfs, _progress_bar)
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
    c = x['total_terminal_fee0'].hist(bins=30, backend='plotly')
    c.update_layout(showlegend=False)
    c2 = x['total_equity_valuation0'].hist(bins=30, backend='plotly')
    c2.update_layout(showlegend=False)
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        with st.expander("Management Fee Distribution", expanded=True):
            st.plotly_chart(c, use_container_width=True)

    with dist_col2:
        with st.expander("Equity Value Distribution", expanded=True):
            st.plotly_chart(c2, use_container_width=True)




def build_sim_metrics(x):
    global values
    incubation_success_ratio_values = [list(DIST_SELECTION.keys())[i] for i in values]
    st.subheader("Valuation Summary")
    total_invested = x['total_invested'].mean()
    PE_ratio = 10.5 
    valuation_man_fees_total: List[float] = [x[f'total_terminal_fee{i}'].mean() * PE_ratio for i in range(4)]
    valuation_equity_total: List[float] = [x[f'total_equity_valuation{i}'].mean() for i in range(4)]
    variable_expenses = INVESTMENT_EXPENSES_PER_DEAL * VALUATION_PERIOD * DEAL_FREQ
    opex_total = sum([OPERATIONAL_EXPENSES*pow((1 + OPEX_INFLATION), i) for i in range(int(VALUATION_PERIOD))])
    net_invested: List[float] = [(valuation_man_fees_total[i] + valuation_equity_total[i] - (x['total_invested'].mean() + variable_expenses + opex_total)) for i in range(4)]
    x["total_valuation_value"] = (x['total_terminal_fee0'] * PE_ratio) + x['total_equity_valuation0'] - (x['total_invested'] + variable_expenses + opex_total)

    import plotly.graph_objects as go

    pie_labels = ['Guardrisk Earnings <br> Contribution', 'Launchpad Equity <br> Contribution']
    pie_values = [valuation_man_fees_total[0], valuation_equity_total[0]]
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

    net_valuation_incl_all_cycles_discounted = sum([net_invested[i]*pow(1+DISCOUNT_RATE, -i*5) for i in range(4)])
    with st.expander(f"Valuation Total (4 Cycles of {VALUATION_PERIOD} years each)", expanded =True):
        st.markdown(f"<h4 style='text-align: center; color: #FC766AFF; font-weight: lighter; padding-bottom: 0'>LP Valuation Contribution to GR</h4>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: #FC766AFF ;padding-top: 0; padding-bottom: 3%; font-weight: 200'>{human_format(net_valuation_incl_all_cycles_discounted)}</h1>", unsafe_allow_html=True)
        # st.metric(f"NPV in Year {VALUATION_PERIOD}", net_valuation_incl_all_cycles_discounted)
    with st.expander("Headline Valuation per Investment Cycle (5 Year Cycles, Not Discounted)", expanded=True):
        cyc_col1, cyc_col2, cyc_col3, cyc_col4 = st.columns(4)
        with cyc_col1:
            st.metric(f"Cycle 1 (Year 1 to {VALUATION_PERIOD})", human_format(net_invested[0]))
            st.caption(f"Incubation Success: {incubation_success_ratio_values[0]}")
        with cyc_col2:
            st.metric(f"Cycle 2 (Year {VALUATION_PERIOD+1} to {VALUATION_PERIOD*2})", human_format(net_invested[1]))
            st.caption(f"Incubation Success: {incubation_success_ratio_values[1]}")
        with cyc_col3:
            st.metric(f"Cycle 3 (Year {VALUATION_PERIOD*2+1} to {VALUATION_PERIOD*3})", human_format(net_invested[2]))
            st.caption(f"Incubation Success: {incubation_success_ratio_values[2]}")
        with cyc_col4:
            st.metric(f"Cycle 4 (Year {VALUATION_PERIOD*3+1} to {VALUATION_PERIOD*4})", human_format(net_invested[3]))
            st.caption(f"Incubation Success: {incubation_success_ratio_values[3]}")

    with st.expander("Headline Net Valuation Result (Cycle 1)", expanded=True):
        st.metric("Launchpad + Guardrisk Value", human_format(net_invested[0]))
        col1, _, col3 = st.columns(3)
        with col1:
            fig
        with col3:
            st.markdown("<h4 style='color: #FC766AFF; font-weight: lighter; padding-bottom: 0'>Totals (Cycle 1)</h4>", unsafe_allow_html=True)
            st.metric("Earnings Contribution", human_format(valuation_man_fees_total[0]))
            st.metric("Portfolio Equity Valuation", human_format(valuation_equity_total[0]))
            st.metric("Amount Invested", '(' + human_format(total_invested) + ')')
            st.metric("Operational Expenses", '(' + human_format(opex_total) + ')')
            st.metric("Variable Expenses", '(' + human_format(variable_expenses) + ')')
        st.markdown(f"<h5 style='text-align: center; color: #FC766AFF; font-weight: lighter; padding-bottom: 0'>Equity Cash-on-Cash Return Multiple</h5>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #FC766AFF ;padding-top: 0; padding-bottom: 3%; font-weight: 200'>{int(valuation_equity_total[0]/total_invested)}X</h2>", unsafe_allow_html=True)
        try:
            dist_graph_total_value = x['total_valuation_value'].hist(bins=30, backend='plotly')
        except:
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
            st.metric("Total Successes", f"{int(round(x['successes'].mean(),0))}")
        with col_summary8:
            st.metric("Total Failures", f"{int(round(x['failures'].mean(),0))}")


int(round(14.7,0))
def main():
    ...


if __name__ == "__main__":
    dfs = build_distribution_table()
    _progress_bar = st.progress(0)
    sim_result_data = build_simulation_result_tables(dfs, _progress_bar)
    build_sim_metrics(sim_result_data)
    build_distribution_charts(sim_result_data)
    
    
