import scrapy   # however, when you pip install, you should install Scrapy

class QuotesSpider(scrapy.Spider):
    name = 'quotes'

    #explicit way
    def start_requests(self):
        urls = [
            'http://quotes.toscrape.com/page/1/',
            'http://quotes.toscrape.com/page/2/',
        ]
        for url in urls:
            yield scrapy.Request(url=url,callback=self.parse)

    # # implicit way
    # start_urls = [
    #     'http://quotes.toscrape.com/page/1/',
    #     'http://quotes.toscrape.com/page/2/',
    # ]
    

    # # save the whole page
    # def parse(self,response):
    #     page = response.url.split('/')[-2]
    #     filename = 'quotes-%s.html' % page
    #     with open(filename,'wb') as f:
    #         f.write(response.body)
    #     self.log('Saved file %s' % filename)

    def parse(self,response):
        for quote in response.css('div.quote'):
            yield {
                'text': quote.css('span.text::text').get(),
                'author': quote.css('small.author::text').get(),
                'tags': quote.css('div.tags a.tag::text').getall(),
            }