import json
import requests
import pickle
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import base64
import io


from tools import normalize_title, get_intersection

app = Flask(__name__)


top_collections = {
    "Bored Ape Yacht Club": "0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13D",
    "CryptoPunks": "0xb47e3cd837dDF8e4c57F05d70Ab865de6e193BBB",
    "Mutant Ape Yacht Club": "0x60E4d786628Fea6478F785A6d7e704777c86a7c6",
    "Mutant Hound Collars": "0xaE99a698156eE8F8d07Cbe7F271c31EeaaC07087",
    "Yaypegs": "0x7fda36c8daEDcc55B73E964c2831D6161eF60a75",
    "Terraforms by Mathcastles": "0x4E1f41613c9084FdB9E34E11fAE9412427480e56",
    "Otherdeed for Otherside": "0x34d85c9CDeB23FA97cb08333b511ac86E1C4E258",
    "Wolf Game": "0x7F36182DeE28c45dE6072a34D29855BaE76DBe2f",
    "Bored Ape Kennel Club": "0xba30E5F9Bb24caa003E9f2f0497Ad287FDF95623",
    "Valhalla": "0x231d3559aa848Bf10366fB9868590F01d34bF240",
    "oraand": "0xF21Ab54111EB5049e42f4794D2724658Baa42FC1",
    "Savage Nation": "0x61E3D1C26802DF805e9Fc22Dc26342e29eCFe860",
    "Chimera King": "0x6950f7Ec392911De504A79F5334D39F4933fAF25",
    "Wired Beast": "0x2910312A1e3e4cdf3c33dFBc88d1e1d7a22e2Bbf",
    "Project SHINOBIS": "0xc64efC58fb656f389F0F7C25C92B7ecF8D02D740"
}



@app.route('/')
def home():
    return render_template("index.html")


@app.route('/result', methods=['POST', 'GET'])
def result():
    wallet = request.form['wallet']

    url = "https://nfts-by-address.p.rapidapi.com/getNFTs/"

    querystring = {"owner": wallet}

    headers = {
        "X-RapidAPI-Key": "88c43860e4msh21604eaaee966bbp1b353ajsn2ac05435c2ea",
        "X-RapidAPI-Host": "nfts-by-address.p.rapidapi.com"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    collection_data = json.loads(response.text)
    collection_data = collection_data['ownedNfts']
    wallet_collections = []
    for i in range(0, len(collection_data)):
        wallet_collections.append(collection_data[i]['title'])

    for i in range(0, len(wallet_collections)):
        wallet_collections[i] = normalize_title(wallet_collections[i])


    inter = get_intersection(wallet_collections, top_collections.keys())
    y_scores = []
    x_scores = []
    for top_collection in top_collections.keys():
        if top_collection in wallet_collections:
            continue
        # print(f'model_1/{top_collection}.pkl' )
        pickled_model = pickle.load(open(f'models/{top_collection}.pkl', 'rb'))
        y_scores.append(pickled_model.predict_proba(inter.drop(top_collection, axis=1))[0][1])
        x_scores.append(top_collection)

    y_mx = 0
    for y in y_scores:
        y_mx = max(y_mx, y)
    for i in range(0, len(y_scores)):
        y_scores[i] /= y_mx

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#131415')
    ax.bar(x=x_scores, height=y_scores, color='#F1BA0D')

    ax.xaxis.set_tick_params(rotation=70)  # Rotating the xticks for 70 degree
    ax.set_title('Probability preference coefficient')
    ax.set_facecolor("#2c2c30")

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")

    sorted_xy = {}
    for i in range(0, len(x_scores)):
        sorted_xy[x_scores[i]] = y_scores[i]

    sorted_xy = sorted(sorted_xy.items(), key=lambda x: x[1], reverse=True)
    sorted_xy = dict(sorted_xy)

    return render_template('result.html', data=f"data:image/png;base64,{data}", x = list(sorted_xy.keys()),
                           y = list(sorted_xy.values()), len=len(sorted_xy))


if __name__ == "__main__":
    app.run(debug=True)
