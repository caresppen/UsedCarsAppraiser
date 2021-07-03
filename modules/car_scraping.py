import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from lxml import etree, html
import re

# base_used_cars_url = 'https://www.coches.com/coches-segunda-mano/coches-ocasion.htm?page='
# base_km0_cars_url = 'https://www.coches.com/km0/coches-seminuevos.htm?page='
# base_renting_cars_url = 'https://www.coches.com/renting-coches/ofertas-renting/?page='

def webpages_generator(base_url, init_pages, n_pages):
    '''
    Generates a list with all the links
    * base_url = <str> common string on the url for each page
    * init_pages = <int> number of the first page available in the web
    * n_pages = <int> number of the last page available in the web
    '''
    all_urls = []
    for page in range(init_pages, n_pages):
        all_urls.append(base_url + str(page))
    
    return all_urls



def cars_links_generator(input_urls, reg_exp):
    '''
    Links list generator of each car
    * input_urls = <list> web urls that contain several car links
    * reg_exp = <str> regular expression to identify the desired links
    '''
    output_urls = []
    
    for url in input_urls:
        
        # Finding websites: coches-segunda-mano; km0; renting
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
                    
        for link in soup.find_all(attrs={'href': re.compile(reg_exp)}):
            output_urls.append(link.get('href'))
            
    return output_urls



def scrape_used_cars_data(input_urls):
    '''
    Scraper function of each car
    Returns a nested list with the parameters of each car
    * input_urls = <list> used/km0 cars urls to be scrapped
    '''
    nested_list = [[]]
    
    for url in input_urls:
        
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        dom = etree.HTML(str(soup))

        try:
            # general data
            title = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardPromotion"]/div[1]/div[1]/h1/strong')[0].text.strip())
            price = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardPrice"]')[0].text.strip())
            year = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardYear"]')[0].text.strip())
            kms = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardKms"]')[0].text.strip())
            city = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardPromotion"]/div[1]/div[1]/div/div')[0].text.strip())
            gear = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardTransmission"]')[0].text.strip())
            doors = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardDoors"]')[0].text.strip())
            seats = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardTechnicalData"]/div[1]/div/div[6]/div[2]')[0].text.strip())
            power = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardPower"]')[0].text.strip())
            color = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardColour"]')[0].text.strip())
            co2_emiss = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardEmission"]')[0].text.strip())
            fuel_type = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardFuel"]')[0].text.strip())
            warranty = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardPpal"]/div[1]/div[4]/div[1]/div[10]/div[2]')[0].text.strip())
            dealer = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardDealerType"]')[0].text.strip())

            # technical data
            chassis = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardChassis"]')[0].text.strip())
            height = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardTechnicalData"]/div[1]/div/div[3]/div[2]')[0].text.strip())
            length = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardTechnicalData"]/div[1]/div/div[2]/div[2]')[0].text.strip())
            width = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardTechnicalData"]/div[1]/div/div[4]/div[2]')[0].text.strip())
            trunk_vol = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardTechnicalData"]/div[1]/div/div[1]/div[2]')[0].text.strip())
            max_speed = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardTechnicalData"]/div[2]/div/div[1]/div[2]')[0].text.strip())

            urban_cons = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardTechnicalData"]/div[2]/div/div[3]/div[2]')[0].text.strip())
            xtrurban_cons = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardTechnicalData"]/div[2]/div/div[4]/div[2]')[0].text.strip())
            mixed_cons = re.sub(r'\s+',' ', dom.xpath('//*[@id="trackingIndexCardConsumption"]')[0].text.strip())

            weight = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardTechnicalData"]/div[1]/div/div[8]/div[2]')[0].text.strip())
            tank_vol = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardTechnicalData"]/div[1]/div/div[7]/div[2]')[0].text.strip())
            acceleration = re.sub(r'\s+',' ', dom.xpath('//*[@id="indexCardTechnicalData"]/div[2]/div/div[5]/div[2]')[0].text.strip())

            car_info = [title, price, year, kms, city, gear, doors, seats, power, color, co2_emiss, fuel_type, warranty, dealer, chassis,
                        height, length, width, trunk_vol, max_speed, urban_cons, xtrurban_cons, mixed_cons, weight, tank_vol, acceleration]
        
            nested_list.append(car_info)
            # print(car_info)
        
        except Exception as e:
            print("Exception raised: {}".format(e))
        
    return nested_list



def scrape_renting_cars_data(input_urls):
    '''
    Scraper function of each renting car
    Returns a nested list with the parameters of each car
    * input_urls = <list> renting cars urls to be scrapped
    '''
    nested_list = [[]]
    
    for url in input_urls:
        
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        dom = etree.HTML(str(soup))

        try:
            # general data
            title = re.sub(r'\s+',' ', dom.xpath('/html/body/section[2]/div[1]/div[1]/h1/span[1]')[0].text.strip()) + ' '\
                    + re.sub(r'\s+',' ', dom.xpath('/html/body/section[2]/div[1]/div[1]/h1/span[2]')[0].text.strip())
            price = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[2]/div/div[4]/div[1]/span[2]')[0].text.strip())
            contract_months = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[2]/div/div[1]/p/span[1]')[0].text.strip())
            km_year = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[2]/div/div[2]/p/span[1]')[0].text.strip())
            
            # Specs
            fuel_type = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[3]/div/div[1]/div[2]')[0].text.strip())
            color = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[3]/div/div[2]/div[2]')[0].text.strip())
            warranty = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[3]/div/div[3]/div[2]')[0].text.strip())
            maintenance = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[3]/div/div[4]/div[2]')[0].text.strip())
            tires = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[3]/div/div[5]/div[2]')[0].text.strip())
            power = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[3]/div/div[6]/div[2]')[0].text.strip())
            co2_emiss = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[3]/div/div[7]/div[2]')[0].text.strip())
            doors = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[1]/div/div[5]/div[2]')[0].text.strip())
            gear = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[3]/div/div[9]/div[2]')[0].text.strip())
            status = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[3]/div/div[10]/div[2]')[0].text.strip())

            # technical data
            chassis = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[1]/div/div[10]/div[2]')[0].text.strip())
            height = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[1]/div/div[3]/div[2]')[0].text.strip())
            length = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[1]/div/div[2]/div[2]')[0].text.strip())
            width = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[1]/div/div[4]/div[2]')[0].text.strip())
            trunk_vol = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[1]/div/div[1]/div[2]')[0].text.strip())
            max_speed = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[2]/div/div[1]/div[2]')[0].text.strip())
            seats = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[1]/div/div[6]/div[2]')[0].text.strip())

            urban_cons = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[2]/div/div[3]/div[2]')[0].text.strip())
            xtrurban_cons = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[2]/div/div[4]/div[2]')[0].text.strip())
            mixed_cons = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[2]/div/div[2]/div[2]')[0].text.strip())

            weight = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[1]/div/div[8]/div[2]')[0].text.strip())
            tank_vol = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[1]/div/div[7]/div[2]')[0].text.strip())
            acceleration = re.sub(r'\s+',' ', dom.xpath('//*[@id="rentingIndexCardInfo"]/div[6]/div[2]/div/div[5]/div[2]')[0].text.strip())

            car_info = [title, price, contract_months, km_year, fuel_type, color, warranty, maintenance, tires, power, co2_emiss, doors, gear, status,
                        chassis, height, length, width, trunk_vol, max_speed, seats, urban_cons, xtrurban_cons, mixed_cons, weight, tank_vol, acceleration]
            
            nested_list.append(car_info)
            # print(car_info)
            
        except Exception as e:
            print("Exception raised: {}".format(e))
        
    return nested_list
