import time
import requests
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import re
import numpy as np
from tabulate import tabulate
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.ensemble import VotingRegressor
from bs4 import BeautifulSoup
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
import streamlit as st


st.set_page_config(layout="wide", page_title="Dorecast")

end = datetime.now() + timedelta(days=1)


startdate = datetime.now() - timedelta(days=41)
startdate = startdate.strftime("%Y-%m-%d")

def send_telegram_message(message):
    bot_token = '6490925202:AAFoJrRj8l428Q1P8czlUfcoeTEF0dFlbZ4'
    chat_id = '@dorecast'
    message = str(message)
    st.write(message)

    cleaned_message = re.sub(r'\[|\]', '', message)
    cleaned_message2 = re.sub(r'\'|\,', '', cleaned_message)
    table = tabulate([cleaned_message2.split(',')], headers="firstrow")
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&text={table}"

    response = requests.get(url)
    return response
def send_telegram_file(file_path):
    bot_token = '6490925202:AAFoJrRj8l428Q1P8czlUfcoeTEF0dFlbZ4'
    chat_id = '@dorecast'
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"

    with open(file_path, 'rb') as file:
        response = requests.post(url, data={'chat_id': chat_id}, files={'document': file})

    return response


def bisto():
    # Web sayfasının URL'si
    url = 'https://www.kap.org.tr/tr/bist-sirketler'

    # Web sayfasını iste
    response = requests.get(url)
    html_content = response.content

    # BeautifulSoup ile HTML içeriğini parse et
    soup = BeautifulSoup(html_content, 'html.parser')

    # "comp-cell _04 vtable" sınıfına sahip elemanları bul
    comp_cells = soup.find_all('div', class_='comp-cell _04 vtable')

    # Her bir "comp-cell _04 vtable" elemanının altındaki <a> etiketlerinin içeriğini al
    text_list = []
    for cell in comp_cells:
        links = cell.find_all('a')
        for link in links:
            text_list.append(link.text)

    # Sonucu yazdır
    return text_list

bist_list=bisto()
bist=[]

for item in bist_list:
    parts = item.split(',')  # Split each item by comma
    if len(parts) >= 1:
        first_part = parts[0].strip()  # Get the first part and remove any leading/trailing spaces
        first_part = first_part[0].upper() + first_part[1:]  # Convert first character to uppercase
        bist.append(first_part + ".IS")

cr=['BTC-USD', 'ETH-USD', 'BNB-USD', 'BCC-USD', 'NEO-USD', 'LTC-USD', 'QTUM-USD', 'ADA-USD', 'XRP-USD', 'EOS-USD', 'TUSD-USD', 'IOTA-USD', 'XLM-USD', 'ONT-USD', 'TRX-USD', 'ETC-USD', 'ICX-USD', 'VEN-USD', 'NULS-USD', 'VET-USD', 'PAX-USD', 'BCHABC-USD', 'BCHSV-USD', 'USDC-USD', 'LINK-USD', 'WAVES-USD', 'BTT-USD', 'USDS-USD', 'ONG-USD', 'HOT-USD', 'ZIL-USD', 'ZRX-USD', 'FET-USD', 'BAT-USD', 'XMR-USD', 'ZEC-USD', 'IOST-USD', 'CELR-USD', 'DASH-USD', 'NANO-USD', 'OMG-USD', 'THETA-USD', 'ENJ-USD', 'MITH-USD', 'MATIC-USD', 'ATOM-USD', 'TFUEL-USD', 'ONE-USD', 'FTM-USD', 'ALGO-USD', 'USDSB-USD', 'GTO-USD', 'ERD-USD', 'DOGE-USD', 'DUSK-USD', 'ANKR-USD', 'WIN-USD', 'COS-USD', 'NPXS-USD', 'COCOS-USD', 'MTL-USD', 'TOMO-USD', 'PERL-USD', 'DENT-USD', 'MFT-USD', 'KEY-USD', 'STORM-USD', 'DOCK-USD', 'WAN-USD', 'FUN-USD', 'CVC-USD', 'CHZ-USD', 'BAND-USD', 'BUSD-USD', 'BEAM-USD', 'XTZ-USD', 'REN-USD', 'RVN-USD', 'HC-USD', 'HBAR-USD', 'NKN-USD', 'STX-USD', 'KAVA-USD', 'ARPA-USD', 'IOTX-USD', 'RLC-USD', 'MCO-USD', 'CTXC-USD', 'BCH-USD', 'TROY-USD', 'VITE-USD', 'FTT-USD', 'EUR-USD', 'OGN-USD', 'DREP-USD', 'TCT-USD', 'WRX-USD', 'BTS-USD', 'LSK-USD', 'BNT-USD', 'LTO-USD', 'STRAT-USD', 'AION-USD', 'MBL-USD', 'COTI-USD', 'STPT-USD', 'WTC-USD', 'DATA-USD', 'XZC-USD', 'SOL-USD', 'CTSI-USD', 'HIVE-USD', 'CHR-USD', 'GXS-USD', 'ARDR-USD', 'LEND-USD', 'MDT-USD', 'STMX-USD', 'KNC-USD', 'REP-USD', 'LRC-USD', 'PNT-USD', 'COMP-USD', 'BKRW-USD', 'SC-USD', 'ZEN-USD', 'SNX-USD', 'VTHO-USD', 'DGB-USD', 'GBP-USD', 'SXP-USD', 'MKR-USD', 'DAI-USD', 'DCR-USD', 'STORJ-USD', 'MANA-USD', 'AUD-USD', 'YFI-USD', 'BAL-USD', 'BLZ-USD', 'IRIS-USD', 'KMD-USD', 'JST-USD', 'SRM-USD', 'ANT-USD', 'CRV-USD', 'SAND-USD', 'OCEAN-USD', 'NMR-USD', 'DOT-USD', 'LUNA-USD', 'RSR-USD', 'PAXG-USD', 'WNXM-USD', 'TRB-USD', 'BZRX-USD', 'SUSHI-USD', 'YFII-USD', 'KSM-USD', 'EGLD-USD', 'DIA-USD', 'RUNE-USD', 'FIO-USD', 'UMA-USD', 'BEL-USD', 'WING-USD', 'UNI-USD', 'NBS-USD', 'OXT-USD', 'SUN-USD', 'AVAX-USD', 'HNT-USD', 'FLM-USD', 'ORN-USD', 'UTK-USD', 'XVS-USD', 'ALPHA-USD', 'AAVE-USD', 'NEAR-USD', 'FIL-USD', 'INJ-USD', 'AUDIO-USD', 'CTK-USD', 'AKRO-USD', 'AXS-USD', 'HARD-USD', 'DNT-USD', 'STRAX-USD', 'UNFI-USD', 'ROSE-USD', 'AVA-USD', 'XEM-USD', 'SKL-USD', 'SUSD-USD', 'GRT-USD', 'JUV-USD', 'PSG-USD', '1INCH-USD', 'REEF-USD', 'OG-USD', 'ATM-USD', 'ASR-USD', 'CELO-USD', 'RIF-USD', 'BTCST-USD', 'TRU-USD', 'CKB-USD', 'TWT-USD', 'FIRO-USD', 'LIT-USD', 'SFP-USD', 'DODO-USD', 'CAKE-USD', 'ACM-USD', 'BADGER-USD', 'FIS-USD', 'OM-USD', 'POND-USD', 'DEGO-USD', 'ALICE-USD', 'LINA-USD', 'PERP-USD', 'RAMP-USD', 'SUPER-USD', 'CFX-USD', 'EPS-USD', 'AUTO-USD', 'TKO-USD', 'PUNDIX-USD', 'TLM-USD', 'BTG-USD', 'MIR-USD', 'BAR-USD', 'FORTH-USD', 'BAKE-USD', 'BURGER-USD', 'SLP-USD', 'SHIB-USD', 'ICP-USD', 'AR-USD', 'POLS-USD', 'MDX-USD', 'MASK-USD', 'LPT-USD', 'NU-USD', 'XVG-USD', 'ATA-USD', 'GTC-USD', 'TORN-USD', 'KEEP-USD', 'ERN-USD', 'KLAY-USD', 'PHA-USD', 'BOND-USD', 'MLN-USD', 'DEXE-USD', 'C98-USD', 'CLV-USD', 'QNT-USD', 'FLOW-USD', 'TVK-USD', 'MINA-USD', 'RAY-USD', 'FARM-USD', 'ALPACA-USD', 'QUICK-USD', 'MBOX-USD', 'FOR-USD', 'REQ-USD', 'GHST-USD', 'WAXP-USD', 'TRIBE-USD', 'GNO-USD', 'XEC-USD', 'ELF-USD', 'DYDX-USD', 'POLY-USD', 'IDEX-USD', 'VIDT-USD', 'USDP-USD', 'GALA-USD', 'ILV-USD', 'YGG-USD', 'SYS-USD', 'DF-USD', 'FIDA-USD', 'FRONT-USD', 'CVP-USD', 'AGLD-USD', 'RAD-USD', 'BETA-USD', 'RARE-USD', 'LAZIO-USD', 'CHESS-USD', 'ADX-USD', 'AUCTION-USD', 'DAR-USD', 'BNX-USD', 'RGT-USD', 'MOVR-USD', 'CITY-USD', 'ENS-USD', 'KP3R-USD', 'QI-USD', 'PORTO-USD', 'POWR-USD', 'VGX-USD', 'JASMY-USD', 'AMP-USD', 'PLA-USD', 'PYR-USD', 'RNDR-USD', 'ALCX-USD', 'SANTOS-USD', 'MC-USD', 'ANY-USD', 'BICO-USD', 'FLUX-USD', 'FXS-USD', 'VOXEL-USD', 'HIGH-USD', 'CVX-USD', 'PEOPLE-USD', 'OOKI-USD', 'SPELL-USD', 'UST-USD', 'JOE-USD', 'ACH-USD', 'IMX-USD', 'GLMR-USD', 'LOKA-USD', 'SCRT-USD', 'API3-USD', 'BTTC-USD', 'ACA-USD', 'ANC-USD', 'XNO-USD', 'WOO-USD', 'ALPINE-USD', 'T-USD', 'ASTR-USD', 'GMT-USD', 'KDA-USD', 'APE-USD', 'BSW-USD', 'BIFI-USD', 'MULTI-USD', 'STEEM-USD', 'MOB-USD', 'NEXO-USD', 'REI-USD', 'GAL-USD', 'LDO-USD', 'EPX-USD', 'OP-USD', 'LEVER-USD', 'STG-USD', 'LUNC-USD', 'GMX-USD', 'NEBL-USD', 'POLYX-USD', 'APT-USD', 'OSMO-USD', 'HFT-USD', 'PHB-USD', 'HOOK-USD', 'MAGIC-USD', 'HIFI-USD', 'RPL-USD', 'PROS-USD', 'AGIX-USD', 'GNS-USD', 'SYN-USD', 'VIB-USD', 'SSV-USD', 'LQTY-USD', 'AMB-USD', 'BETH-USD', 'USTC-USD', 'GAS-USD', 'GLM-USD', 'PROM-USD', 'QKC-USD', 'UFT-USD', 'ID-USD', 'ARB-USD', 'LOOM-USD', 'OAX-USD', 'RDNT-USD', 'WBTC-USD', 'EDU-USD', 'SUI-USD', 'AERGO-USD', 'PEPE-USD', 'FLOKI-USD', 'AST-USD', 'SNT-USD', 'COMBO-USD', 'MAV-USD', 'PENDLE-USD', 'ARKM-USD', 'WBETH-USD', 'WLD-USD', 'FDUSD-USD', 'SEI-USD', 'CYBER-USD', 'ARK-USD', 'CREAM-USD', 'GFT-USD', 'IQ-USD', 'NTRN-USD', 'TIA-USD', 'MEME-USD', 'ORDI-USD', 'BEAMX-USD', 'PIVX-USD', 'VIC-USD', 'BLUR-USD', 'VANRY-USD', 'AEUR-USD', 'JTO-USD', '1000SATS-USD', 'BONK-USD', 'ACE-USD', 'NFP-USD', 'AI-USD', 'XAI-USD', 'MANTA-USD', 'ALT-USD', 'PYTH-USD', 'RONIN-USD', 'DYM-USD', 'PIXEL-USD', 'STRK-USD', 'PORTAL-USD', 'PDA-USD', 'AXL-USD', 'WIF-USD', 'METIS-USD', 'AEVO-USD', 'BOME-USD', 'ETHFI-USD', 'ENA-USD', 'W-USD', 'TNSR-USD', 'SAGA-USD', 'TAO-USD', 'OMNI-USD', 'REZ-USD', 'BB-USD', 'NOT-USD', 'IO-USD', 'ZK-USD', 'LISTA-USD', 'ZRO-USD', 'G-USD', 'BANANA-USD', 'RENDER-USD', 'TON-USD', 'DOGS-USD']



liste=cr
st.write(str("kripto: "+str(len(cr))))
st.write(str("bist: "+str(len(bist))))

kripto=st.checkbox("kripto",value=True)
bisti=st.checkbox("bist",value=False)
if kripto==True:
    liste=cr
elif bisti==True:
    liste=bist
elif kripto==True and bisti==True:
    liste=cr+bist
else:
    st.warning("seçim yapınız")
def newthe(liste,end=end):
    sepet = pd.DataFrame(columns=["symbol","close","difference",
                                  "forecast","date",
                                  "hedef_tarih","MAPE"])
    interval='1h'


    süre=24*3
    beklenen_sinyal=1
    sell=pd.DataFrame(columns=["symbol","close","date"])
    say1 = 0
    for symbol in liste:
        try:
            say1 += 1


            if symbol.endswith(".IS"):
                beklenen_degisim = 5
            else:
                beklenen_degisim = 10
            st.write(str(say1)+"/"+str(len(liste))+" "+symbol)

            data = yf.download(symbol,threads=True,repair=True,start=startdate, progress=False, interval=interval, end=end)
            data['close'] = data['Close']
            data.reset_index(inplace=True)
            ortalama = data['close'].mean()

            data['Datetime'] = pd.to_datetime(data['Datetime']).dt.tz_localize(None)

            # Teknik göstergeleri hesapla
            def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
                data['EMA12'] = data['close'].ewm(span=short_window, adjust=False).mean()
                data['EMA26'] = data['close'].ewm(span=long_window, adjust=False).mean()
                data['MACD'] = data['EMA12'] - data['EMA26']
                data['Signal Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
                data['MACD Signal'] = np.where(data['MACD'] > data['Signal Line'], 1, -1)

            def calculate_rsi(data, window=14):
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                data['RSI'] = 100 - (100 / (1 + rs))
                data['RSI Signal'] = np.where(data['RSI'] < 30, 1, np.where(data['RSI'] > 70, -1, 0))

            def calculate_kdj(data, window=14):
                low_min = data['Low'].rolling(window=window).min()
                high_max = data['High'].rolling(window=window).max()
                data['%K'] = 100 * ((data['close'] - low_min) / (high_max - low_min))
                data['%D'] = data['%K'].rolling(window=3).mean()
                data['%J'] = 3 * data['%K'] - 2 * data['%D']
                data['KDJ Signal'] = np.where(data['%J'] < 20, 1, np.where(data['%J'] > 80, -1, 0))

            def calculate_EMA(data):
                data['EMA20'] = data['close'].ewm(span=20, adjust=False).mean()
                data['EMA120'] = data['close'].ewm(span=120, adjust=False).mean()
                data['EMA Signal'] = np.where(data['EMA20'] > data['EMA120'], 1, -1)

            def calculate_indicators(data):
                calculate_macd(data)
                calculate_rsi(data)
                calculate_EMA(data)
                calculate_kdj(data)

            def generate_signals(data):
                data['Signal'] = data[['MACD Signal', 'KDJ Signal', 'RSI Signal', 'EMA Signal']].sum(axis=1)

            # Teknik göstergeleri hesapla ve sinyalleri oluştur
            calculate_indicators(data)
            generate_signals(data)

            if data['Signal'].iloc[-1] >= 1:
                data.fillna(0, inplace=True)

                def create_lagged_features(data, lag=süre):
                    df = data.copy()
                    for i in range(1, lag + 1):
                        df[f'lag_{i}'] = df['close'].shift(i)
                    return df

                data_with_lags = create_lagged_features(data)
                data_with_lags.dropna(inplace=True)
                X = data_with_lags.drop(['Datetime', 'close', 'Signal Line', 'MACD Signal', 'RSI Signal',
                                         'KDJ Signal', 'EMA Signal', 'Signal'], axis=1)
                y = data_with_lags['close']
                X_train, X_test, y_train, y_test = X.iloc[:-süre], X.iloc[-süre:], y.iloc[:-süre], y.iloc[-süre:]

                estimators1 = [10, 100, 50, 150, 200]
                for estimators in estimators1:
                    models = {
                        'rf': RandomForestRegressor(n_estimators=estimators, random_state=42),
                        'gb': GradientBoostingRegressor(n_estimators=estimators, learning_rate=0.1, max_depth=3,
                                                        random_state=42),
                        'ab': AdaBoostRegressor(n_estimators=estimators, learning_rate=1.0, random_state=42),
                        'xgb': XGBRegressor(n_estimators=estimators, learning_rate=0.1, max_depth=6, random_state=42,
                                            verbosity=0),
                        'lgbm': LGBMRegressor(n_estimators=estimators, learning_rate=0.05, random_state=42,
                                              verbosity=-1)
                    }

                # Modelleri ve performanslarını depolamak için bir sözlük
                model_performance = {}

                # Modelleri deneyin ve performanslarını değerlendirin
                for name, model in models.items():
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
                    model_performance[name] = np.mean(np.sqrt(-scores))

                # Performansları karşılaştırın
                sorted_models = sorted(model_performance.items(), key=lambda x: x[1])
                best_models = sorted_models[:2]

                # Seçilen modellerle ensemble modeli oluşturun
                best_model_objects = [(name, models[name]) for name, _ in best_models]
                ensemble_model = VotingRegressor(estimators=best_model_objects)

                # Ensemble modelini eğitin
                ensemble_model.fit(X_train, y_train)

                future_predictions = []
                last_known_features = X_test.iloc[-1].values
                for i in range(süre):
                    prediction = ensemble_model.predict(last_known_features.reshape(1, -1))[0]
                    future_predictions.append(prediction)

                    # Gecikmeli özellikleri güncelleyin
                    last_known_features = np.roll(last_known_features, -1)
                    last_known_features[-1] = prediction

                y_pred = ensemble_model.predict(X_test)
                beklenen_MAPE = 10
                MAPE=np.mean(np.abs((y_test - y_pred) / y_test)) * 100

                seventh_day_prediction = future_predictions[-1]
                if MAPE > beklenen_MAPE:
                    st.write("BEKLENEN MAPE: " + str(beklenen_MAPE))
                    st.write(symbol+ " YÜKSEK MAPE: " + str(MAPE))
                elif MAPE <= beklenen_MAPE:

                    if seventh_day_prediction > data[['close']].tail(1).values[0][0]:
                        difference_percent=((seventh_day_prediction-data[['close']].tail(1).values[0][0])/data[['close']].tail(1).values[0][0])*100
                        if difference_percent>beklenen_degisim:
                            st.write("Değişim oranı: " + str(difference_percent) + "%")
                            st.write(symbol)
                            hedef_tarih=(datetime.now()+timedelta(days=süre)).strftime("%d.%m.%Y")
                            add_data=pd.DataFrame({'symbol':[symbol],
                                                   'close':[data[['close']].tail(1).values[0][0]],
                                                   'difference':[difference_percent],
                                                   'forecast':[seventh_day_prediction],
                                                   'date':[datetime.now().strftime("%d.%m.%Y")],
                                                   'hedef_tarih':[hedef_tarih],
                                                   'MAPE':[MAPE]})
                            add_data = add_data.dropna(axis=1, how='all')
                            send_telegram_message(symbol+" değişim : "+str(difference_percent)+" ")
                            sepet=pd.concat([sepet,add_data],ignore_index=True)

                        else:
                             st.write("Değişim oranı yetersiz: " + str(difference_percent) + "%" )
                    else:
                         st.write( symbol+" yetersiz .....değişim : "+str(seventh_day_prediction-data[['close']].tail(1).values[0][0]))
                else:
                    st.write("BEKLENEN MAPE: " + str(beklenen_MAPE))
                    st.write(symbol + " YÜKSEK MAPE: " + str(MAPE))
            elif data['Signal'].iloc[-1] <=-3:
                st.write(symbol + " Düşecek!!! " )

                add_data_sell=pd.DataFrame({'symbol':[symbol],'close':[data[['close']].tail(1).values[0][0]],'date':[datetime.now().strftime("%d.%m.%Y")]})
                sell = sell.dropna(axis=1, how='all')
                add_data_sell = add_data_sell.dropna(axis=1, how='all')
                sell=pd.concat([sell,add_data_sell],ignore_index=True)
            else:
                st.write(symbol + " sinyal gücü!  : " +str(data['Signal'].iloc[-1]))

        except Exception as e:
            st.write(str(e))

    file='buy_list.xlsx'
    old_data=pd.read_excel(file)
    sepet=sepet.sort_values(by='difference', ascending=False)
    sepet.dropna()
    old_data.dropna()
    if sepet.empty:
        buylist = old_data.copy()  # Avoid modifying old_data
    else:
        # Check for columns with only NA values in sepet (optional)
        na_cols = sepet.columns[sepet.isna().all()]
        if len(na_cols) > 0:
            sepet.drop(na_cols, axis=1, inplace=True)  # Drop NA columns (optional)
            st.dataframe(sepet)
        buylist = pd.concat([old_data, sepet], ignore_index=True)
    send_telegram_message("Buy sinyali verenler: " )
    send_telegram_message(str(sepet))
    send_telegram_message("Sell sinyali verenler: " + str(sell))
    #make an excel
    df = pd.DataFrame(buylist)
    df.sort_values(by='difference', ascending=False)
    df.to_excel('buy_list.xlsx',index=False,engine='openpyxl')
    dfsell=pd.DataFrame(sell)
    dfsell.to_excel('sell_list.xlsx')
    send_telegram_file('sell_list.xlsx')
    send_telegram_file('buy_list.xlsx')


if st.button("Analiz"):
    while True:
        baslangıc=datetime.now()
        st.write("başlangıç:"+str(baslangıc))
        newthe(liste)
        st.write("bitti...")
        st.write("Analiz süresi:",str(datetime.now()-baslangıc))
        send_telegram_message("Analiz süresi: " + str(datetime.now()-baslangıc))
        time.sleep(60*60)

