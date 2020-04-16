import requests
import json

client_id = "b137bf0a83b75afff254"
client_secret = "2710d9a68d2a0f52ba8fc6460eef1bfd18094a61"
token = "f01d9ed0ab21001233c8ff58c7566338412dd416"
base = "https://api.github.com/"
get_contents = "repos/moevm/scientific_writing-2019/contents/"
PARAMS = {'note': "somethings",
          'client_id': 'justaleaf',
          'client_secret': token}
HEADERS = {'Authorization': "token " + token}


def downloadFile(URL, file_name):
    import urllib.request
    import shutil
    with urllib.request.urlopen(URL) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)


r = requests.get(url=base + get_contents, headers=HEADERS, params=PARAMS)
papers_download_urls = []
count = 0  # всего папок в мастере
counter = 0  # всего статей
for obj in r.json():
    re = requests.get(url=base + get_contents + obj["name"], headers=HEADERS, params=PARAMS)
    count += 1
    print(obj["name"])
    for re_obj in re.json():
        if type(re_obj) == str:
            continue
        print(re_obj)
        if re_obj["name"] == 'paper.md':
            counter += 1
            #papers_download_urls.append({'download_url': re_obj["download_url"]})
            downloadFile(re_obj["download_url"], "papers/"+ str(counter))

print(count)
print(counter)
#with open('sw_papers.json', 'w') as f:
#    json.dump(papers_download_urls, f, ensure_ascii=False)
