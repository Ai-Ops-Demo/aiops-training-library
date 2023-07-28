from dash import Dash, dcc, html, Input, Output,ctx
import plotly.express as px
import pandas as pd
import os
import json

app = Dash(__name__)


#directory to data
directory = 'data/norm_data/'
files = os.listdir(directory)

jsonname = 'timestamps.json'
val_jsonname = 'val.json'

#get column names from a random file
defaultfile = files[0]
columns = pd.read_csv(directory+defaultfile).columns


#create a json file if one doesnt exist
if not os.path.exists(jsonname):
	with open(jsonname, 'w') as outfile:
				json.dump({'0':''}, outfile,indent=4)

#create a json file if one doesnt exist
if not os.path.exists(val_jsonname):
	with open(val_jsonname, 'w') as outfile:
				json.dump({'0':''}, outfile,indent=4)


app.layout = html.Div(
	children = [
		dcc.Dropdown(
			id='file',
			options=files,
			value=defaultfile,
			),
		dcc.Dropdown(
			id='tags',
			options=columns,
			multi=True,
			value=[],
			),
		dcc.Graph(
			id="vars",
			figure={}
			),

		html.Button('Save Training', id='save', n_clicks=0),
		html.Button('Save Validation', id='save_val', n_clicks=0),
		html.Div(id='save_msg',children=[])
		]
	)

#graph and drop down callbacks
@app.callback(
    Output(component_id="vars",component_property='figure'), 
    [Input(component_id="tags",component_property='value'),
	Input(component_id="file",component_property='value'),]
	)
def update_line_chart(tags,file):
	data = pd.read_csv('data/norm_data/'+file)
	graph_df = pd.concat((data['TimeStamp'],data[tags]),axis=1)
	graph_df['TimeStamp'] = pd.to_datetime(data['TimeStamp'])
	fig = px.line(
		graph_df, 
		x='TimeStamp',
		y=tags,
		height = 700, 
		width = 1800,
		template = "plotly_dark",)

	return fig

#save function callback
@app.callback(
	Output('save_msg', 'children'),
	[Input('save', 'n_clicks'),
  	Input('save_val', 'n_clicks'),
	Input('vars', 'relayoutData'),
	Input(component_id="file",component_property='value')]
	)

def savefile(n_clicks1,n_clicks2,relayoutData,file):
	msg = "Save plot data"
	if 'save' == ctx.triggered_id:
		#open json
		with open(jsonname, 'r') as savefile:
			savedict = json.load(savefile)
		
		#get next record
		records = list(savedict.keys())
		lastrecord = int(records[-1])

		xmin = relayoutData['xaxis.range[0]']
		xmax = relayoutData['xaxis.range[1]']
		savedict[str(lastrecord+1)] = {
			'file': file,
			'xmin': xmin,
			'xmax': xmax
			}

		#delete the 0th entry because its blank
		if '0' in savedict:
			del savedict['0']

		#save file
		with open(jsonname, 'w') as outfile:
			json.dump(savedict, outfile,indent=4)

		msg = 'Training record saved ' + str(file) + '  Range: ' + xmin + ' to ' + xmax

	elif 'save_val' == ctx.triggered_id:
		
		#open json
		with open(val_jsonname, 'r') as savefile:
			savedict = json.load(savefile)
		
		#get next record
		records = list(savedict.keys())
		lastrecord = int(records[-1])

		xmin = relayoutData['xaxis.range[0]']
		xmax = relayoutData['xaxis.range[1]']
		savedict[str(lastrecord+1)] = {
			'file': file,
			'xmin': xmin,
			'xmax': xmax
			}

		#delete the 0th entry because its blank
		if '0' in savedict:
			del savedict['0']

		#save file
		with open(val_jsonname, 'w') as outfile:
			json.dump(savedict, outfile,indent=4)

		msg = 'Validation record saved ' + str(file) + '  Range: ' + xmin + ' to ' + xmax

	return html.Div(msg)

if __name__ == '__main__':
	app.run_server(debug=True)