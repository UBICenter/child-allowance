import pandas as pd
import numpy as np
import microdf as mdf
import plotly.express as px
import plotly.graph_objects as go

person_raw = pd.read_csv('https://github.com/ngpsu22/2016-2018-ASEC-/raw/master/cps_00004.csv.gz')

person = person_raw.copy(deep=True)
person.columns = person.columns.str.lower()
person = person.drop(['serial', 'month', 'pernum', 'cpsidp', 'asecwth'], axis=1)
person = person.rename(columns={'asecwt':'weight','statefip': 'state'})

person['state'] = person['state'].astype(str)
person['state'].replace({'1':'Alabama','2':'Alaska', '4': 'Arizona','5':'Arkansas',
                         '6': 'California', '8': 'Colorado', '9': 'Connecticut',
                         '10':'Delaware', '11': 'District of Columbia', '12':'Florida',
                         '13': 'Georgia','15':'Hawaii', '16':'Idaho','17':'Illinois',
                         '18':'Indiana', '19':'Iowa','20':'Kansas', '21': 'Kentucky',
                         '22':'Louisiana', '23': 'Maine', '24': 'Maryland',
                         '25':'Massachusetts', '26':'Michigan', '27': 'Minnesota',
                         '28':'Mississippi','29':'Missouri', '30': 'Montana',
                         '31': 'Nebraska', '32':'Nevada', '33': 'New Hampshire',
                         '34': 'New Jersey', '35': 'New Mexico', '36':'New York',
                         '37':'North Carolina', '38':'North Dakota', '39': 'Ohio',
                         '40':'Oklahoma', '41': 'Oregon', '42':'Pennsylvania',
                         '44':'Rhode Island','45':'South Carolina', '46':'South Dakota',
                         '47': 'Tennessee', '48':'Texas','49':'Utah','50':'Vermont',
                         '51':'Virginia', '53':'Washington', '54':'West Virginia',
                         '55':'Wisconsin', '56':'Wyoming'},inplace=True)

person['child'] = person.age < 18
person['adult'] = person.age >= 18
ages = person.groupby(['spmfamunit','year'])[['child','adult']].sum()
ages.columns = ['total_children', 'total_adults']
person = person.merge(ages,left_on=['spmfamunit', 'year'], right_index=True)

def ca_pov(state, age_group, ca_monthly=0):
  target_persons = person[person.state==state].copy(deep=True)

  if age_group == 'child':
    target_persons = target_persons[target_persons.child]
  if age_group == 'adult':
    target_persons = target_persons[target_persons.adult]
  
  target_persons['total_ca'] = target_persons.total_children * ca_monthly * 12
  target_persons['new_spm_resouces'] = target_persons.total_ca + target_persons.spmtotres
  target_persons['poor'] = target_persons.new_spm_resouces < target_persons.spmthresh
  target_pop = (target_persons.weight).sum()
  total_poor = (target_persons.weight * target_persons.poor).sum()

  return (total_poor / target_pop * 100).round(1)

def pov_row(row):
  return ca_pov(row.state, row.age_group, row.ca_monthly)

summary = mdf.cartesian_product({'state':person.state.unique(),
                       'ca_monthly': np.arange(0,501,25),
                       'age_group': ['child', 'adult', 'all']})

summary['poverty_rate'] = summary.apply(pov_row, axis=1)
summary = summary.sort_values(['state', 'ca_monthly'], ascending= (True, True)) 

def line_graph(df, x, y, color, title, xaxis_title, yaxis_title):
    fig = px.line(df, x=x, y=y, color=color)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis_ticksuffix='%',
        font=dict(family='Roboto'),
        hovermode='x', 
        xaxis_tickprefix='$',
        xaxis_ticksuffix='',
        plot_bgcolor='white',
        legend_title_text=''   
    )

    fig.update_traces(mode='markers+lines', hovertemplate=None)

    fig.show()

summary2 = summary[~summary.age_group.isin(['all', 'adult'])]
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
dc = {'code':'DC', 'state': 'District of Columbia'}
df = df.append(dc, ignore_index=True)
df = (df.merge(summary2, left_on='state', right_on='state'))
df = df.filter(['code','state', 'ca_monthly', 'poverty_rate'], axis=1)
# df.to_csv(r'state_25.csv')

map = px.choropleth(df, 
              locations = 'code',
              color="poverty_rate", 
              animation_frame="ca_monthly",
              color_continuous_scale="reds",
              locationmode='USA-states',
              scope="usa",
              range_color=(0, 24),
              title='',
              height=600,
              labels={'ca_monthly': "Monthly Child Allowance",
                      'code':'State',
                      'poverty_rate': 'Child Poverty Rate'
                    }
             )
map

raw_data = summary
state_names = raw_data['state'].unique()
state_names.sort()
age_groups = raw_data['age_group'].unique()
x = raw_data['ca_monthly'].unique()

data_list = []
for state in state_names:
  state_list = []
  state_data = raw_data[raw_data['state']==state]
  for age in age_groups:
    state_list.append(state_data[state_data['age_group']==age]['poverty_rate'])
  data_list.append(state_list)

data = pd.DataFrame(data_list, columns = age_groups)
data['State'] = state_names
data = data.set_index('State')

fig = go.Figure()

legend_names = {'child': 'Child poverty',
                'adult': 'Adult poverty',
                'all': 'Overall poverty'}
default = state_names[0]
for age in age_groups:
  fig.add_trace(go.Scatter(
      x=x, 
      y=data[age][default],
      name=legend_names[age]
    ))

buttons = []
title = 'The Poverty Impact of a Child Allowance in '
for state in state_names:
  new_button = {'method': 'update',
                'label': state,
                'args': [{'y': data.loc[state]}, {'title.text': title + state}]}
  buttons.append(new_button)

# construct menus
updatemenus = [{'buttons': buttons,
                'direction': 'down',
                'showactive': True,}]

# update layout with buttons, and show the figure
fig.update_layout(updatemenus=updatemenus)

fig.update_layout(
      title= title + default,
      xaxis_title='Monthly Child Allowance',
      yaxis_title='SPM poverty rate',
      yaxis_ticksuffix='%',
      font=dict(family='Roboto'),
      hovermode='x', 
      xaxis_tickprefix='$',
      xaxis_ticksuffix='',
      plot_bgcolor='white',
      legend_title_text=''   
)

fig.update_layout(
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99
))

fig.update_traces(mode='markers+lines', hovertemplate=None)

fig.update_layout(
    xaxis = dict(
        tickmode = 'linear',
        dtick = 50
    )
)

fig

