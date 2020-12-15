'''
pip install Scrapy
pip install selenium

In a folder:
    scrapy startproject imgt
when running:
    scrapy crawl imgt -o out.json
when using scrapy shell:
    scrapy shell 'url'


conda activate selenium

remember make change to the python scirpt under spider folder
'''


import scrapy
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

class imgtSpider(scrapy.Spider):
    name = 'imgt'
    start_urls = ['http://www.imgt.org/3Dstructure-DB/']

    pyramid = {
        8: {1:(1),2:(2),3:(3),4:(4),5:(5),6:(6),7:(7),8:(8)},
        9: {1:(1,2),2:(2,3),3:(3,4),4:(4,5),5:(5,6),6:(6,7),7:(7,8),8:(8,9)},
        10: {1:(1,2,3),2:(2,3,4),3:(3,4,5),4:(4,5,6),5:(5,6,7),6:(6,7,8),7:(7,8,9),8:(8,9,10)},
        11: {1:(1,2,3,4),2:(2,3,4,5),3:(3,4,5,6),4:(4,5,6,7),5:(5,6,7,8),6:(6,7,8,9),7:(7,8,9,10),8:(8,9,10,11)}
    }

    def __init__(self):
        self.hla = 'HLA-A*9253'
        path_to_chromedriver = '/Users/ligk2e/Downloads/chromedriver'   
        self.driver = webdriver.Chrome(executable_path=path_to_chromedriver)
        self.driver.implicitly_wait(5)

    def get_selenium(self,url):
        self.driver.get(url)
        self.driver.find_element_by_xpath('//*[@id="species"]/option[27]').click()    # choose Home Sapien   (select drop down)
        self.driver.find_element_by_xpath('//*[@id="radio_pMH1"]').click()            # choose pMHCI   (input)
        self.driver.find_element_by_xpath('//*[@id="datas"]/p[2]/input[1]').click()   # click submit    (button)
        return self.driver.page_source.encode('utf-8')


    def parse(self,response):   # for parsing 549 entry page
        response = scrapy.Selector(text=self.get_selenium(imgtSpider.start_urls[0]))

        for row in response.css('body#result div#data table.Results tbody tr')[1:]:  #[Selector,Selector,Selector...]   # don't need header
            mhc = row.css('td')[2].css('td::text').get()

            if self.hla in mhc:
                # yield {
                #     'mhc':mhc
                # }
                url_suffix = row.css('td')[1].css('a::attr(href)').get()      # details.cgi?pdbcode=2CLR
                # what we need is: http://www.imgt.org/3Dstructure-DB/cgi/details.cgi?pdbcode=2CLR&Part=CONT_OVERVIEW
                url_next = 'http://www.imgt.org/3Dstructure-DB/cgi/' + url_suffix + '&Part=CONT_OVERVIEW'
                yield scrapy.Request(url_next,callback=self.parse1)

    def parse1(self,response):   # for parsing each PDB entry page for contact analysis
        include_next = False
        for row in response.css('body#result div#mybody div#main table.contacts tbody tr'):
            if include_next:
                url_suffix = row.css('td')[0].css('a')[-1].css('a::attr(href)').get()
                url_next = 'http://www.imgt.org/3Dstructure-DB/cgi/' + url_suffix
                yield scrapy.Request(url_next,callback=self.parse_alpha2)
                include_next = False
            else:
                if len(row.css('td')) == 1:  # gap between two blocks
                    continue
                else:
                    identity = row.css('td')[2].css('td::text').get()
                    if identity == '(Ligand)':
                        include_next = True
                        url_suffix = row.css('td')[0].css('a')[-1].css('a::attr(href)').get()
                        url_next = 'http://www.imgt.org/3Dstructure-DB/cgi/' + url_suffix
                        yield scrapy.Request(url_next,callback=self.parse_alpha1)

    def parse_alpha1(self,response):
        url = response.url
        table = response.xpath('//*[@id="main"]/table[3]')
        length = table.css('tr')[-1].css('td')[1].css('td::text').get()
        length = int(length)
        #yield {'test':length}
        if int(length) in set([8,9,10,11]):
            dic = {}
            for row in table.css('tr')[3:]:   # remove some headers
                ligand, hla = row.css('td')[1].css('td::text').get(), row.css('td')[7].css('td::text').get()
                try:
                    ligand, hla = int(ligand), int(hla)
                except ValueError:
                    continue
                try:
                    dic[ligand].append(hla)
                except KeyError:
                    dic[ligand] = []
                    dic[ligand].append(hla)   # ligand position 1 interact with hla position 5 dict[1] = 5
            if length == 8: 
                dic_alpha1 = dic
            else:
                pyramid_table = imgtSpider.pyramid[length]
                dic_alpha1 = {}
                for key,value in pyramid_table.items():
                    bucket = []
                    for i in value:   #(1,2)
                        try:
                            bucket.extend(dic[i])
                        except KeyError:    # some cases, not every position is interacting with hla
                            continue
                    dic_alpha1[key] = bucket
            yield {
                'alpha1_interaction_{}'.format(url):dic_alpha1,
            }
        else:
            yield {
                'alpha1_interaction_{}'.format(url):'This ligand is not from 8-11 mer'
            }
                






    
    def parse_alpha2(self,response):
        url = response.url
        table = response.xpath('//*[@id="main"]/table[3]')
        length = table.css('tr')[-1].css('td')[1].css('td::text').get()
        length = int(length)
        #yield {'test':length}
        if int(length) in set([8,9,10,11]):
            dic = {}
            for row in table.css('tr')[3:]:   # remove some headers
                ligand, hla = row.css('td')[1].css('td::text').get(), row.css('td')[7].css('td::text').get()
                try:
                    ligand, hla = int(ligand), int(hla)
                except ValueError:
                    continue
                try:
                    dic[ligand].append(hla)
                except KeyError:
                    dic[ligand] = []
                    dic[ligand].append(hla)   # ligand position 1 interact with hla position 5 dict[1] = 5
            if length == 8: 
                dic_alpha2 = dic
            else:
                dic_alpha2 = {}
                pyramid_table = imgtSpider.pyramid[length]
                for key,value in pyramid_table.items():
                    bucket = []
                    for i in value:   #(1,2)
                        try:
                            bucket.extend(dic[i])
                        except KeyError:    # some cases, not every position is interacting with hla
                            continue

                    dic_alpha2[key] = bucket
            yield {
                'alpha2_interaction_{}'.format(url):dic_alpha2,
            }
        else:
            yield {
                'alpha2_interaction_{}'.format(url):'This ligand is not from 8-11 mer'
            }



    



