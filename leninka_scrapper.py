#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import urllib
from urllib.request import urlopen, urlretrieve
from bs4 import BeautifulSoup as BS
import os
import sys
import glob
import json

disciplineLinks = ["https://cyberleninka.ru/article/c/computer-and-information-sciences/"]
baseLink = "https://cyberleninka.ru"
baseFolder = "papers/"
baseName = "paper"
baseJournalFolder = "journal"
baseJournalName = "journal"

counter = 1
papersList = []
for disc in disciplineLinks:
    for page in range(1, 10):
        print("page #" + str(page))
        html = urlopen(disc + str(page))
        print(disc + str(page))
        soup = BS(html, features="lxml")
        elems = [baseLink + x['href'] + '/pdf' for x in soup.findAll('a') if x['href'].find("article/n/") != -1]
        #print(elems[2][:-4])
        #links = [baseLink + y['href'] for y in soup.findAll('a') if y['href'].find("journal/n/") != -1]

        for elem in elems:
            html2 = urlopen(elem[:-4])
            soup2 = BS(html2, features="lxml")
            if soup2.select_one('#body > div.content > div > span > div:nth-child(2) > h1 > i') is None:
                print("stop")
                break
            paperTitle = soup2.select_one('#body > div.content > div > span > div:nth-child(2) > h1 > i').text
            paperText = soup2.find("div", {"itemprop": "articleBody"}).text
            print(paperTitle)
            journal = [baseLink + y['href'] for y in soup2.findAll('a') if y['href'].find("journal/n/") != -1]
            html3 = urlopen(journal[0])
            soup3 = BS(html3, features="lxml")
            title = soup3.findAll('h1')[0].text
            statItems = [x.text for x in soup3.findAll("div", {"class": "statitem"})]

            paperObj = {
                'journalName': title,
                'journalViews':  int(statItems[0]),
                'journalDownloads': int(statItems[1]),
                'journalHirch': int(statItems[2]),
                'paperPath': baseFolder + baseName + str(counter),
                'paperUrl': elem[:-4],
                'paperTitle': paperTitle,
                'paperRawText': paperText
            }
            papersList.append(paperObj)

            fileInfo = urlretrieve(elem, baseFolder + baseName + str(counter) + ".pdf")
            #if os.stat(fileInfo[0]).st_size < 15000: #file less than appr. 15kb
            #    sys.exit("Captcha!")
            counter+=1

sortedPapersList = sorted(papersList, key=lambda paper: int(paper['journalHirch']), reverse=True)
with open('rawdata.json', 'w', encoding='utf-8') as f:
    json.dump(papersList, f, ensure_ascii=False)
#with open('sorteddata.json', 'w', encoding='utf-8') as f:
#    json.dump(sortedPapersList, f, ensure_ascii=False)



