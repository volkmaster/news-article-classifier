#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Ziga Vucko'


import scrapy


class WikiSpider(scrapy.Spider):
    name = 'wiki'
    start_urls = []
    url = 'https://en.wikipedia.org/wiki/'
    months = ['January', 'February', 'March', 'April', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    start_year, end_year = 1998, 2015
    for month in months:
        for year in range(start_year, end_year+1):
            start_urls.append(url + month + '_' + str(year))

    def parse(self, response):
        days = response.xpath('//table[@class="vevent"]')
        for day in days:
            description = day.xpath('.//td[@class="description"]')
            event_types = description.xpath('dl/dt/text()').extract()
            # no event type specified
            if len(event_types) == 0:
                continue
            categories = description.xpath('ul')
            for i, category in enumerate(categories):
                urls = category.xpath('.//li/a[@class="external text"]/@href').extract()
                for url in urls:
                    yield {
                        'url': url,
                        'event_type': event_types[i]
                    }
