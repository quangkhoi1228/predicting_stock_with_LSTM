import requests

#the required first parameter of the 'get' method is the 'url':
x = requests.get('https://chart.aladin.finance/history?symbol=VNINDEX&resolution=D&from=1599709676&to=1600573676')

#print the response text (the content of the requested file):
print(x.text)
