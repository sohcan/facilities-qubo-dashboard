from os import mkdir

import pandas as pd
import numpy as np

from dwave.samplers import SimulatedAnnealingSampler
from pyqubo import Binary, LogEncInteger, Placeholder, Constraint
from pathlib import Path

import streamlit as st
import altair as alt

st.set_page_config(
    page_title="Facilities Project Optimization",
    page_icon=":material/fort:",
    initial_sidebar_state='expanded',
    layout="wide"
)

if 'lim_Bp' not in st.session_state:
    st.session_state['lim_Bp'] = 2300000

if 'lim_Bs' not in st.session_state:
    st.session_state['lim_Bs'] = 1800000

if 'Project_Type' not in st.session_state:
    st.session_state['Project_Type'] = 'Lab_Type_A'

try:
    HERE = Path(__file__).parent
except NameError:
    HERE = Path()

DATA = HERE / "data_synthetic"
DATO = HERE / "data_synthetic_out"

DATO.mkdir(exist_ok=True)

WBS_csv = "WBS_Funding_Splits.csv"
FIM_csv ="FIMS_Master_Scenario_Cost.csv"
QUBO_csv ="Result_QUBO.csv"
FIMS_csv = "FIMS_Inventory.csv"
CAIS_csv = "CAIS_Deficiencies.csv"

print(f"HERE: {HERE}")
print(f"DATA: {DATA}")

#DATA / WBS_csv
#DATA / CAIS_csv
#DATO / QUBO_csv
#DATO / EXPS_csv
#DATO / EXPS_all_csv

FIM_dat = pd.read_csv( (DATA / FIM_csv))
WBS_dat = pd.read_csv((DATA / WBS_csv))
CAIS_sim_dat = pd.read_csv((DATA / CAIS_csv))
FIMS_dat = pd.read_csv((DATA / FIMS_csv))

Project_Type = st.session_state['Project_Type']
cst_Project_Type = st.session_state['Project_Type'] + '_Cost'

Non_Project_Type = 'Non_' + Project_Type
Non_cst_Project_Type = 'Non_' + cst_Project_Type

CAIS_sim_dat = CAIS_sim_dat.merge(FIMS_dat[['Property_ID','WBS_Element']], on='Property_ID', how='left')
CAIS_sim_dat = CAIS_sim_dat.merge(WBS_dat[['WBS_Code','Fed_Type_A','Fed_Type_B','Lab_Type_A']], left_on='WBS_Element', right_on='WBS_Code', how='left').rename(columns={'Correction_Cost':'Cost'}).drop(columns={'WBS_Code'})
CAIS_dat = CAIS_sim_dat[['WBS_Element','Property_ID',Project_Type,'Cost']].groupby(['WBS_Element','Property_ID',Project_Type]).sum().reset_index()

CAIS_dat[Project_Type + '_%'] = (CAIS_dat[Project_Type] / 100)
CAIS_dat[Non_Project_Type + '_%'] = (1 - CAIS_dat[Project_Type + '_%'])
CAIS_dat[cst_Project_Type] = (CAIS_dat.Cost * CAIS_dat[Project_Type + '_%'])
CAIS_dat[Non_cst_Project_Type] = (CAIS_dat.Cost * CAIS_dat[Non_Project_Type + '_%'])



x = {index : Binary(f"x_{index}") for index in CAIS_dat.index}
l = {index : f"x_{index}" for index in CAIS_dat.index}
CAIS_dat['Qx'] = x
CAIS_dat['Lx'] = l

###########################################SHAPES
CAIS_dat_cols = CAIS_dat.shape[1]
scenario_cols = CAIS_sim_dat.shape[1]
experiment_count = 5

#COST FUNCTIONS
cst_p = CAIS_dat[cst_Project_Type] @ CAIS_dat.Qx
cst_s = CAIS_dat[Non_cst_Project_Type] @ CAIS_dat.Qx

#REWARDS
A_P = Placeholder('A_P')
A_S = Placeholder('A_S')

#PENALTIES
M_P = Placeholder('M_P')
M_S = Placeholder('M_S')
lambda_P = Placeholder('lambda_P')
lambda_S = Placeholder('lambda_S')

#PARAMTERS
lim_Bp = st.session_state['lim_Bp']
lim_Bs = st.session_state['lim_Bs']

#PARAMETERSMODEL
feed_dict = {
    'A_P' : -1,
    'A_S' : -1,
    'M_P' : 10,
    'M_S' : 10,
    'lambda_P' : 0,
    'lambda_S' : 0
}

#CONSTRAINT TUNINGS
slk_P = LogEncInteger("slk_P", (0, lim_Bp))
slk_S = LogEncInteger("slk_S", (0, lim_Bs))

####HERERERERE WEEEE GOOOO))))
E_Qx = A_P*(cst_p) + A_S*(cst_s) + M_P*(cst_p + slk_P - lim_Bp )**2 + M_S*(cst_s + slk_S - lim_Bs)**2 + lambda_P*(slk_P/lim_Bp)**2 + lambda_S*(slk_S/lim_Bs)**2
model = E_Qx.compile()
qubo, offset = model.to_qubo(feed_dict=feed_dict)

sampler = SimulatedAnnealingSampler()
sampleset = sampler.sample_qubo(qubo, num_reads=(num_reads :=100))
samplesetFrame = sampleset.to_pandas_dataframe()

#SSAMPLESET DESCRIPTIONS
max_offset_energy = samplesetFrame['energy'].max()+offset

lin_shots = pd.DataFrame(samplesetFrame['energy']).reset_index(names='Read')
lin_shots['Minimums'] = lin_shots['energy'] == lin_shots['energy'].min()
lin_shots['Log_E'] = lin_shots['energy'] - lin_shots['energy'].min() + .01


CAIS_dat = CAIS_dat.merge(samplesetFrame.sort_values('energy',ascending=True).T.add_prefix('EXP'), how='outer' ,left_on='Lx',right_index=True)
Property_ID_key = CAIS_dat[['Property_ID','Lx']].merge(samplesetFrame.sort_values('energy',ascending=True).T.add_prefix('EXP'), how='outer' ,left_on='Lx',right_index=True)

CAIS_display = CAIS_dat[['Property_ID',cst_Project_Type,Non_cst_Project_Type,'Cost']].dropna(
).style.format(
    {
        'Property_ID': "{:.0f}",
        cst_Project_Type : "${:,.2f}",
        Non_cst_Project_Type : "${:,.2f}",
        'Cost': "${:,.2f}"
        }
    )

QUBOScenario_all = pd.merge(CAIS_sim_dat, Property_ID_key, how="left", on='Property_ID' )

QUBOScenario = QUBOScenario_all.iloc[0:,0:(scenario_cols + 1 + experiment_count )]
QUBOScenario.insert(
    1 + QUBOScenario.columns.get_loc('Cost'),
    'Cost_Primary',
    QUBOScenario['Cost']*QUBOScenario[Project_Type]/100)
QUBOScenario.insert(
    2 + QUBOScenario.columns.get_loc('Cost'),
    'Cost_Share',
    QUBOScenario['Cost']*(100-QUBOScenario[Project_Type])/100)

CAIS_chrt = CAIS_dat.dropna().iloc[:,0:-num_reads]
CAIS_chrt['Frequency'] = CAIS_dat.dropna().iloc[:,-num_reads+1:].sum(axis=1)
CAIS_chrt.drop(columns='Qx', inplace=True)
limit_values = [{'limit':f"{Project_Type} Budget Limit", 'value': lim_Bp},{'limit': f"{Project_Type} Budget B Limit", 'value': lim_Bs}]

#CHARTS----------------->
##########CHART LINE
energy_chart = alt.Chart(lin_shots).encode(x='Read')
energy_line = energy_chart.mark_line().encode(y=alt.Y('Log_E:Q', title= "Log(Energy) + .01", scale=alt.Scale(type='log')))

energy_mins = alt.Chart(lin_shots[lin_shots['Minimums'] == True][['Read','Minimums','Log_E']]).encode(x='Read')
energy_dots = energy_mins.mark_point(size=200,color='red').encode(y='Log_E:Q')

energy_draw = energy_line + energy_dots

#########CHART LEGEND
cost_legend = alt.Legend(title='Cost Components', orient='top' )
acct_legend = alt.Legend(title='Budget Limits', orient='top' )
acct_scale = alt.Scale(domain=[x['limit'] for x in limit_values], range=['red','grey'])

#######CHART A
sort = alt.EncodingSortField('Frequency',order='descending')

freq_chart = alt.Chart(CAIS_chrt).encode(x = alt.X('Property_ID:N', sort=sort))
freq_bar = freq_chart.mark_bar().encode(y = alt.Y('Frequency:Q'))

cost_chart = alt.Chart(CAIS_chrt).transform_fold(fold=[cst_Project_Type, Non_cst_Project_Type], as_=['Entity','Total'])
cost_bar = cost_chart.mark_bar().encode(x = alt.X('Property_ID:N', sort=sort), y=alt.Y('Total:Q'), color=alt.Color('Entity:N', legend=cost_legend))
lim_chart = alt.Chart(alt.Data(values=limit_values)).mark_rule(size=1).encode(y='value:Q', stroke=alt.Color('limit:N', title='Limits', scale=acct_scale, legend=None))

freq_draw = freq_bar
cost_draw =  cost_bar + lim_chart

#######CHART B
sort2 = alt.EncodingSortField('Cost',order='descending')
flip2 = alt.Axis(orient='right')

freq_chart2 = alt.Chart(CAIS_chrt).encode(x = alt.X('Property_ID:N', sort=sort2))
freq_bar2 = freq_chart2.mark_bar().encode(y = alt.Y('Frequency:Q', axis=flip2))

cost_chart2 = alt.Chart(CAIS_chrt).transform_fold(fold=[cst_Project_Type, Non_cst_Project_Type], as_=['Entity','Total'])
cost_bar2 = cost_chart2.mark_bar().encode(x = alt.X('Property_ID:N', sort=sort2), y=alt.Y('Total:Q', axis=flip2), color=alt.Color('Entity:N', legend=None))

lim_chart2 = alt.Chart(alt.Data(values=limit_values)).mark_rule(size=1).encode(y='value:Q', stroke=alt.Color('limit:N', title='Limits', scale=acct_scale, legend=acct_legend))

freq_draw2 = freq_bar2
cost_draw2 = cost_bar2 + lim_chart2

##################PAGEITUP
st.dataframe(CAIS_chrt)
st.title('QUBO')
st.subheader('Facility Condition Repair Optimization')

with st.sidebar:
    with st.container(border=True):
        st.selectbox('CHANGE TYPE', options = ['Fed_Type_A','Fed_Type_B','Lab_Type_A'],  key = 'Project_Type')
        with st.form('Constraint', border=False):
           st.slider(Project_Type + ' Budget', 1, 50000000, step=100000, key='lim_Bp', format="$%,d")
           st.slider(Non_Project_Type + ' Budget', 1, 50000000, step=100000, key='lim_Bs', format="$%,d")
           st.form_submit_button('New Constraints')

    with st.container(border=True):
        st.markdown(f"""
            OPTIMIZATION RESULT\n
            Min Energy: {sampleset.first.energy}\n
            Occurrences: {sampleset.first.num_occurrences}\n
            Offset (*C*): {offset}\n
            Min + C: {samplesetFrame['energy'].min()+offset}\n
            Max + C: {(walrus := samplesetFrame['energy'].max()+offset)}\n
            Range: {(samplesetFrame['energy'].max()) - (samplesetFrame['energy'].min())}\n 
            Range Log: {np.log10( abs(samplesetFrame['energy'].max() - samplesetFrame['energy'].min()) )}\n
            Walrus: {walrus:.2}\n
            Log(Walrus): {np.log(abs(walrus))/np.log(np.log(abs(walrus)))}\n
            """)


with st.container(border=True):
    st.markdown('Energy Reads')
    st.altair_chart(energy_draw.interactive())

with st.container(border=True):
    st.markdown('Binary Variables: (*x=1*) Frequency, Associated Cost')
    left, right = st.columns(2)
    with left:
        st.altair_chart(cost_draw.interactive())
        st.altair_chart(freq_draw.interactive())
    with right:
        st.altair_chart(cost_draw2.interactive())
        st.altair_chart(freq_draw2.interactive())


with st.container(border=True):
    st.markdown('Binary Variable Outcomes (Reassociated to Project Costs)')
    EXP_cols = st.columns(3, border=False, gap='medium')
    for exp in range(0,experiment_count):
        with EXP_cols[divmod(exp,3)[1]]:
            with st.container(border=False, gap="xxsmall"):

                col_idx = QUBOScenario.iloc[:, -(experiment_count)  + exp]
                col_idx_nom = QUBOScenario.iloc[:, -(experiment_count)  + exp].name
                QUBO_Display_Prime = QUBOScenario[QUBOScenario[col_idx_nom] == 1].groupby(['WBS_Element',Project_Type])[['Cost_Primary','Cost_Share']].sum().reset_index().rename(columns={'Cost_Primary':'Cost'})
                QUBO_Display_Share = QUBOScenario[QUBOScenario[col_idx_nom] == 0].groupby(['WBS_Element',Project_Type])[['Cost_Primary']].sum().reset_index().rename(columns={'Cost_Primary':'Cost'})
                
                Prime_Vars = f'''{Project_Type} Execution: -${QUBO_Display_Prime['Cost'].sum():,.2f}
{Project_Type} Balance: ${lim_Bp-QUBO_Display_Prime['Cost'].sum():,.2f}
Cost Share Execution: -${QUBO_Display_Prime['Cost_Share'].sum():,.2f}
Cost Share Balance: ${lim_Bs-QUBO_Display_Prime['Cost_Share'].sum():,.2f}'''
                
                Share_Vars = f''' Excluded: ${QUBO_Display_Share['Cost'].sum():,.2f}'''


                st.markdown(f'Read: {col_idx_nom}', text_alignment="center")
                with st.expander(label='Included' ):
                    st.table(QUBOScenario[QUBOScenario[col_idx_nom] == 1].groupby(['Property_ID'])['Cost'].sum().reset_index(), hide_index=True, border=False)

                st.code(Prime_Vars)
                st.dataframe(
                    QUBO_Display_Prime,
                    column_config={'Cost_Share': st.column_config.NumberColumn(format="$ %,.2f"), 'Cost': st.column_config.NumberColumn(format="$ %,.0f"), Project_Type:st.column_config.NumberColumn(format="%.0f%%", width='small')},
                    hide_index=True
                    )

                
                with st.expander(f'Excluded'):
                    st.table(QUBOScenario[QUBOScenario[col_idx_nom] == 0].groupby(['Property_ID'])['Cost'].sum().reset_index(), hide_index=True, border=False)

                st.code(Share_Vars)
                st.dataframe(
                    QUBO_Display_Share,
                    column_config={'Cost_Share': st.column_config.NumberColumn(format="$ %,.2f"), 'Cost': st.column_config.NumberColumn(format="$ %,.0f"), Project_Type:st.column_config.NumberColumn(format="%.0f%%", width='small')},
                    hide_index=True
                    )
                st.divider()


st.markdown('Map Selections Back to Uniformat II Systems')
st.dataframe(QUBOScenario)
CAIS_dat.to_csv( (DATO / QUBO_csv) )
