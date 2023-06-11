import argparse
import bs4
import datetime
import gzip
import json
import os
import re
import requests
import pandas as pd
import time as tm
import uuid
from urllib.request import urlopen
from urllib.error import HTTPError
from tqdm import tqdm

NAROU_URL = "https://api.syosetu.com/novelapi/api/"
NAROU18_URL = "https://api.syosetu.com/novel18api/api/"

parser = argparse.ArgumentParser(
            description="generate chat dataset using Narou API",
            add_help=True,
            )
parser.add_argument("--interval4error",
                    default=120,
                    help="wait interval sec for narou API. over 1 sec is highly recommended for safety")
parser.add_argument("--interval",
                    default=2,
                    help="wait interval sec for narou API. over 1 sec is highly recommended for safety")
parser.add_argument("--min_kaiwaritu",
                    default=50,
                    type=int,
                    help="minimum kaiwaritu percentage")
parser.add_argument("--is_narou_18",
                    action="store_true",
                    help="using 18kin api or not")
parser.add_argument("--output_path",
                    default="./output",
                    help="using 18kin api or not")
parser.add_argument("--retry_count",
                    default=5,
                    help="retry count for narou API")
args = parser.parse_args()

def format_text(t):
    return re.sub(r"[\．_－―─＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉・,\./【】→←○《》≪≫\r\u3000\u2000]+", "", t)


def generate_api_url(is_narou_18:bool, genre:int = 102, min_kaiwaritu:int = -1) -> str:
    api_url = NAROU_URL if not is_narou_18 else NAROU18_URL

    options:list = []
    if genre > 0:
        options.append("genre=%d"%genre)

    if min_kaiwaritu > 0:
        options.append("kaiwaritu=%d-"%min_kaiwaritu)
    
    if options:
        api_url += "?"
        for option in options:
            api_url += option+"&"
        api_url = api_url[:-1]

    return api_url


def create_beautifulsoup(url, interval):
    tm.sleep(interval)
    return bs4.BeautifulSoup(urlopen(url).read().decode("utf-8", "ignore"), "html.parser")


def get_bodies_from_url(args, url) -> list:
    try:
        bsobj = create_beautifulsoup(url, args.interval)
    except HTTPError:
        return []

    bodies = []
    for body in bsobj.findAll("div", {"id": "novel_honbun"})[0].findAll("p"):
        formatted = format_text(body.get_text())
        if formatted == "":
            continue
        bodies.append(formatted)
    return bodies


def get_narou_allcount(url) -> int:
    payload = {
        "out": "json",
        "gzip": 5,
        "of": "n",
        "lim":1
    }
    res = requests.get(url, params=payload).content
    r =  gzip.decompress(res).decode("utf-8") 
    allcount = json.loads(r)[0]["allcount"]
    return allcount


def enumerate_all_ncode(args) -> tuple[str, str]:
    api_url: str = generate_api_url(args.is_narou_18, min_kaiwaritu=args.min_kaiwaritu)

    allcount = get_narou_allcount(api_url)
    all_queue_cnt = (allcount // 500) + 10
    for i in range(all_queue_cnt):
        payload = {
            "out": "json",
            "gzip": 5,
            "of": "t-n",
            "lim":500,
            "order": "hyoka"
        }
        
        # なろうAPIにリクエスト
        cnt=0
        while cnt < args.retry_count:
            try:
                res = requests.get(api_url, params=payload, timeout=30).content
                break
            except:
                print("Connection Error")
                cnt = cnt + 1
                tm.sleep(args.interval4error) #接続エラーの場合、120秒後に再リクエストする
            
        r =  gzip.decompress(res).decode("utf-8")   
        df = pd.read_json(r).drop(0)
        for _, row in df.iterrows():
            yield row["title"], row["ncode"]

   
def main():
    print("start",datetime.datetime.now())

    for (title, ncode) in enumerate_all_ncode(args):
        print(title, ncode)
        dump_path = os.path.join(args.output_path, "%s.json"%ncode)
        if os.path.exists(dump_path):
            print(dump_path, "is already exists. skip...")
            continue

        corpus: list = []
        for i in range(1, 1000):
            tm.sleep(args.interval)
            syosetu_url = "https://ncode.syosetu.com/%s/%d/"%(ncode.lower(), i)
            print("extract: ", syosetu_url)
            bodies = get_bodies_from_url(args, syosetu_url)
            if not bodies:
                break

            message_tree_id:str = str(uuid.uuid4())
            parent_id:str = message_tree_id
            p = re.compile(r"^「.*」$")
            for body in tqdm(bodies):
                matched = p.match(body)
                if not matched:
                    continue
                message_id:str = str(uuid.uuid4())
                message = matched.group()
                record:dict = {}
                record["message_tree_id"] = message_tree_id
                record["parent_id"] = parent_id
                record["message_id"] = message_id
                record["text_ja"] = body[1:-1]
                corpus.append(record)

                parent_id = message_id
        
        with open(dump_path, encoding="utf-8", mode="w") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=4)
        print("path:", dump_path, "is saved")

    print("end",datetime.datetime.now())

if __name__ == "__main__":
    main()
