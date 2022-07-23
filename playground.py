import tpqoa
api = tpqoa.tpqoa("oanda.cfg")
print(api.account_type)
# api.get_instruments()[0]
#api.get_history('EUR_USD', '2020-01-20', '2020-01-21', '1d', 'M', localize=False)
data = api.get_history(instrument='AAPL',
                       start='2021-07-01',
                       end='2022-05-31',
                       granularity='M15',
                       price='M')
print(data)
