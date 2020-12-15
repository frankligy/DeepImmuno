'''
pip install Scrapy
pip install selenium

In a folder:
    scrapy startproject imgt
when running:
    scrapy crawl new_imgt -o out.json
when using scrapy shell:
    scrapy shell 'url'
    in Ipython, you can use response.xpath or response.css to try out

    object:
    1. selectorlist    if css('a')  and there are a lot of 'a'
    2. selector   it will have css and xpath method
    3. reponse 


conda activate selenium

remember make change to the python scirpt under spider folder
'''


'''
If encounter robot blockage error:

open setting.py and change the robot setting to False

you can specify hla in __init__, and then when call:

scrapy crawl new_imgt -a hla="HLA-A*0101" -o out.json


When encounter dynamic page, use selenium to get the page and pass it to scrapy response object

Double check using both 'inspect' and 'see source code' in a webpage, they can be different
'''


'''
cat inventory_compliant.txt | while read line; do scrapy crawl new_imgt -a hla="$line" -o "./hla_paratope/$line.json"; done


'''

import scrapy
from scrapy.crawler import CrawlerProcess
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait

class imgtSpider(scrapy.Spider):
    name = 'new_imgt'
    start_urls = ['http://www.imgt.org/3Dstructure-DB/']

    def __init__(self,hla):
        self.hla = hla
        path_to_chromedriver = '/Users/ligk2e/Downloads/chromedriver'   
        self.driver = webdriver.Chrome(executable_path=path_to_chromedriver)
        self.driver.implicitly_wait(5)



    def get_selenium(self,url):
        self.driver.get(url)
        self.driver.find_element_by_xpath('//*[@id="species"]/option[27]').click()    # choose Home Sapien   (select drop down)
        self.driver.find_element_by_xpath('//*[@id="radio_pMH1"]').click()            # choose pMHCI   (input)
        self.driver.find_element_by_xpath('//*[@id="datas"]/p[2]/input[1]').click()   # click submit    (button)
        return self.driver.page_source.encode('utf-8')


    def parse(self,response):   # for parsing 550 entry page
        response = scrapy.Selector(text=self.get_selenium(imgtSpider.start_urls[0]))

        for row in response.css('body#result div#data table.Results tbody tr')[1:]:  #[Selector,Selector,Selector...]   # don't need header
            mhc = row.css('td')[2].css('td::text').get()

            if self.hla in mhc:
                url_suffix = row.css('td')[1].css('a::attr(href)').get()      # details.cgi?pdbcode=2CLR
                # what we need is: http://www.imgt.org/3Dstructure-DB/cgi/details.cgi?pdbcode=2CLR&Part=Epitope
                url_next = 'http://www.imgt.org/3Dstructure-DB/cgi/' + url_suffix + '&Part=Epitope'

                yield scrapy.Request(url_next,callback=self.parse_paratope)



    def parse_paratope(self,response):
        url_next = response.url
        paratope = ''
        for i in response.css('body#result div#mybody div#main table')[0].css('tr')[2].css('td')[1].css('span a'):
            aa = i.css('a::text').get()
            paratope += aa
        yield {'{}'.format(url_next):paratope}

# if using process, you can just run a python new_imgt_spider.py
# process = CrawlerProcess()
# process.crawl(imgtSpider)
# process.start()