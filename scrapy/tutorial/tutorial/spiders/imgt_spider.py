import scrapy
from selenium import webdriver

class imgtSpider(scrapy.Spider):
    name = 'imgt'
    start_urls = ['http://www.imgt.org/3Dstructure-DB/']

    def __init__(self):
        path_to_chromedriver = '/Users/ligk2e/Downloads/chromedriver'   
        self.driver = webdriver.Chrome(executable_path=path_to_chromedriver)


    def parse(self,response):
        self.driver.get(response.url)
        self.driver.find_element_by_xpath('//*[@id="species"]/option[27]').click()    # choose Home Sapien   (select drop down)
        self.driver.find_element_by_xpath('//*[@id="radio_pMH1"]').click()            # choose pMHCI   (input)
        self.driver.find_element_by_xpath('//*[@id="datas"]/p[2]/input[1]').click()   # click submit    (button)
        response1 = self.driver.page_source
        yield response.css('title::text').get()


