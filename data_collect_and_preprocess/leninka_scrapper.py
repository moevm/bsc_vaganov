#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from urllib.request import urlopen
from bs4 import BeautifulSoup as BS
import json


class LeninkaScrapper:

    def __init__(self):
        self.disciplineLinks = ["https://cyberleninka.ru/article/c/computer-and-information-sciences/"]
        self.baseLink = "https://cyberleninka.ru"
        self.baseFolder = "/home/woghan/Desktop/ml/leninka_scrapper/papers/"
        self.baseName = "paper"

    def scrap(self, paper_views_range=150, paper_downloads_range=20, pages=10):
        counter = 1
        papersList = []
        for disc in self.disciplineLinks:
            for page in range(1, pages):
                print("page #" + str(page))
                html = urlopen(disc + str(page))
                print(disc + str(page))
                soup = BS(html, features="lxml")
                elems = [self.baseLink + x['href'] + '/pdf' for x in soup.findAll('a') if x['href'].find("article/n/") != -1]

                for elem in elems:
                    html2 = urlopen(elem[:-4])
                    soup2 = BS(html2, features="lxml")
                    if soup2.select_one('#body > div.content > div > span > div:nth-child(2) > h1 > i') is None:
                        print("Can't collect papers, captcha on CyberLeninka")
                        if papersList:
                            print("Create rawdata.json and stop")
                            with open('rawdata.json', 'w', encoding='utf-8') as f:
                                json.dump(papersList, f, ensure_ascii=False)
                            return True
                        else:
                            print("No papers was found, try again later")
                            return False
                    paperTitle = soup2.select_one('#body > div.content > div > span > div:nth-child(2) > h1 > i').text
                    paperViews = soup2.select_one('#body > div.content > div > span > div:nth-child(2) > div.infoblock.authors.visible > div.top-cc > div.statitem.views').text
                    paperDownloads = soup2.select_one('#body > div.content > div > span > div:nth-child(2) > div.infoblock.authors.visible > div.top-cc > div.statitem.downloads').text
                    print(paperTitle)
                    journal = [self.baseLink + y['href'] for y in soup2.findAll('a') if y['href'].find("journal/n/") != -1]
                    html3 = urlopen(journal[0])
                    soup3 = BS(html3, features="lxml")
                    title = soup3.findAll('h1')[0].text
                    statItems = [x.text for x in soup3.findAll("div", {"class": "statitem"})]

                    if int(paperViews) > paper_views_range and int(paperDownloads) > paper_downloads_range:
                        isGood = 1
                    else:
                        isGood = 0
                    print(isGood)
                    paperObj = {
                        'journalName': title,
                        'journalViews':  int(statItems[0]),
                        'journalDownloads': int(statItems[1]),
                        'journalHirch': int(statItems[2]),
                        'paperPath': self.baseFolder + self.baseName + str(counter) + ".pdf",
                        'paperUrl': elem[:-4],
                        'paperTitle': paperTitle,
                        'isGood': isGood
                    }
                    papersList.append(paperObj)
                    counter+=1

        with open('rawdata.json', 'w', encoding='utf-8') as f:
            json.dump(papersList, f, ensure_ascii=False)
        return True




