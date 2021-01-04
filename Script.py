#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:03:22 2020

@author: nishant
"""
"""
List:-
1.Laptops1 2.Smartphones1 3.Smartwatches 4.SmartfitnessBands
5.Wireless Bluetooth Earbuds1 6.Wireless Airpods 7.Wired Earphones 8.Memory Cards 9.Data Cables 
10.Chargers 11.Wireless Headphones1 12.Wired Headphones 13.Power Banks 14.Gaming consoles1
15.Gaming Monitors1 16.DSLR Cameras 17.Home Theatre Systems 18.Lenses and Camera Accesories
19.Bluetooth Wireless Speakers 20.Wired Speakers 21.Soundbars 22.Projectors
23.SetupBoxes 24.Routers 25.Broadband Modems 26.Full HD Smart LED TV1 
27.Tablets1 28.Pen-Drives1 29.Hardrives1  30.SSD's 31.Keyboards 32.CPU's 33.Mouses
34.Printers 35.Laptop and Accessories 36.Smartphone Accessories
37.Binoculars 38.DJI Drone Camera 39.Intel Real Sense Depth Cameras 40.Surveillance and Security Cameras
41.GPS Accessories 42.Car & Vehicle Electronics Accessories 43.Webcams
44.Adapters 45.PC Headsets 46. Microphones 47.DVD's 48. CD Drives 
49.Pens for Graphic Tablets 50.Numeric Keypads 51.Trackballs
52.Cooling Pads 53.Fire Wire Hubs 54.Joysticks 55.Surge Protectors
56.USB Gadgets 57.Portable Media Players 58.Telephones
59.Telepone Accessories 60.Wearable technology Accessories 61.Network Attached Storage
62.Streaming Media Players 63.Blue Ray Players 64.Android Based TV's
65.4K TV's 66.Telescopes 67.360 degree Cameras 68.Spy Cameras 69.Camcorders
70.Data Cards and Dongles 71.Wireless USB 72.Motherboards 73.Graphic Cards
74.Processors 75.Alexa Smart home 76.VR's 77.Smartplugs 78.Smartlocks
79.Smartlights 80.DJ Controllers 81.Calculators 82.Electric ToothBrush 83.Split Air-Conditioners1
84.Window Air-Conditioners 85 Water-Purifiers 86.Induction Cooktops 
87.Espresso Machines 88.Mixer grinders 89.Juicers 90.Food Processors
91.Oven Toaster Grills 92.Microwave Ovens 93.Kettle
94.Irons 95.Air Purifiers1 96.Sewing & Embroidery Machines 97.Vacuum Cleaners
98.Ceiling Fans 99.Wall Fans 100.Exhaust Fans 101.Pedestral Fans 102.Air Coolers1
103.Room heaters 104.Water Heaters 105.Smart AC's 106.Side-By-Side Refrigerators
107.Double/Triple Door Refrigerators 108.Front Load Washing Machines
109.Fully Automatic Top Load Washing Machines 110.Semi Automatic Washing Machines
111.Washers and Dryers 112.Solar Battery & Chargers 113.Single Door Refrigerators
114.Motion Detectors 115.XBOX 116.Playstations 117.SMART SUNGLASSES1
118.Treadmills 119.Electronic Home Safes
"""

# importing the module 
import pandas as pd 
	
# making data frame from the csv file 
dataframe = pd.read_csv("/home/nishant/Wipro_DataScientist/Projects/E-Commerce/Electronics_store.csv",error_bad_lines=False) 
	
# using the replace() method 
dataframe.replace(to_replace ="shrimp", 
				value = "Wireless Bluetooth Earbuds", 
				inplace = True) 
dataframe.replace(to_replace ="almonds", 
				value = "SMART SUNGLASSES", 
				inplace = True) 
dataframe.replace(to_replace ="escalope", 
				value = "Wireless Headphones", 
				inplace = True) 
dataframe.replace(to_replace ="pasta", 
				value = "Smartphones", 
				inplace = True) 
dataframe.replace(to_replace ="fromage blanc", 
				value = "Gaming Monitors", 
				inplace = True) 
dataframe.replace(to_replace ="honey", 
				value = "Gaming consoles", 
				inplace = True) 
dataframe.replace(to_replace ="light cream", 
				value = "Laptops", 
				inplace = True) 
dataframe.replace(to_replace ="chicken", 
				value = "Hardrives", 
				inplace = True) 
dataframe.replace(to_replace ="olive oil", 
				value = "Pen-Drives", 
				inplace = True) 
dataframe.replace(to_replace ="whole wheat pasta", 
				value = "Full HD Smart LED TV", 
				inplace = True) 
dataframe.replace(to_replace ="tomato sauce", 
				value = "Split Air-Conditioners", 
				inplace = True) 
dataframe.replace(to_replace ="ground beef", 
				value = "Air Purifiers", 
				inplace = True)
dataframe.replace(to_replace ="herb & pepper", 
				value = "Air Coolers", 
				inplace = True)
dataframe.replace(to_replace ="mushroom cream sauce", 
				value = "Tablets", 
				inplace = True)
L1=['asparagus', 'oatmeal', 'grated cheese', 'fresh tuna', 
   'white wine', 'bacon', 'chutney', 'pickles', 'oil', 
   'cookies', 'magazines', 'light mayo', 'low fat yogurt', 'mayonnaise', 'burger sauce', 
   'mint', 'spinach', 'barbecue sauce', 'green beans', 'french fries', 'ketchup', 
   'napkins', 'eggs', 'turkey', 'burgers', 'cake', 'frozen vegetables', 'energy bar', 
   'body spray', 'gluten free bar', 'milk', 'cider', 'french wine', 
   'cauliflower', 'vegetables mix', 'avocado', 'salad', 
   'green grapes', 'antioxydant juice', 'cooking oil', 'parmesan cheese', 
   'tomato juice', 'hot dogs', 'meatballs', 'yogurt cake', 'shampoo', 'shallot', 
   'nonfat milk', 'pancakes', 'carrots', 'bug spray', 'chocolate bread', 'sandwich', 
   'rice', 'muffins', 'energy drink', 'strawberries', 'cream', 'babies food', 
   'spaghetti', 'tea', 'chili', 'melons', 'hand protein bar', 
   'strong cheese', 'soda', 'red wine', 'chocolate', 'brownies', 
   'fresh bread', 'salt', 'frozen smoothie', 'whole wheat rice',  
   'bramble', 'candy bars', 'soup', 'eggplant', 'blueberries', 'black tea', 
   'flax seed', 'green tea', 'gums', 'yams', 'ham', 'clothes accessories', 
   'cereals',  'pet food', 'sparkling water', 'pepper', 
   'mint green tea', 'cottage cheese', 'whole weat flour', 'water spray', 
    'butter', 'corn', 'dessert wine', 'toothpaste', 'mashed potato', 
   'extra dark chocolate', 'salmon', 'mineral water', 'zucchini', 'champagne', 
   'tomatoes', 'protein bar']

L2=["Smartwatches", "SmartfitnessBands","Wireless Airpods" ,"Wired Earphones",
 "Memory Cards","Data Cables" ,"Chargers" ,"Wired Headphones","Power Banks","DSLR Cameras",
 "Home Theatre Systems","Lenses and Camera Accesories","Bluetooth Wireless Speakers", 
 "Wired Speakers","Soundbars","Projectors","SetupBoxes", "Routers","Broadband Modems", 
 "SSD's" ,"Keyboards", "CPU's","Mouses","Printers","Laptop and Accessories", 
 "Smartphone Accessories","Binoculars","DJI Drone Camera", "Intel Real Sense Depth Cameras", 
 "Surveillance and Security Cameras","GPS Accessories", "Car & Vehicle Electronics Accessories", 
 "Webcams","Adapters","PC Headsets","Microphones", "DVD's", "CD Drives", 
 "Pens for Graphic Tablets","Numeric Keypads","Trackballs","Cooling Pads","Fire Wire Hubs", 
 "Joysticks", "Surge Protectors","USB Gadgets","Portable Media Players","Telephones",
 "Telepone Accessories","Wearable technology Accessories","Network Attached Storage",
 "Streaming Media Players","Blue Ray Players","Android Based TV's",
"4K TV's", "Telescopes", "360 degree Cameras", "Spy Cameras","Camcorders","Data Cards",
"Dongles","Wireless USB", "Motherboards", "Graphic Cards","Processors","Alexa Smart home", 
 "VR's", "Smartplugs", "Smartlocks","Smartlights","DJ Controllers",
 "Calculators", "Electric ToothBrush", "Window Air-Conditioners"",Water-Purifiers",
 "Induction Cooktops", "Espresso Machines", "Mixer grinders","Juicers","Food Processors",
"Oven Toaster Grills","Microwave Ovens","Kettle","Irons", "Sewing & Embroidery Machines",
"Vacuum Cleaners","Ceiling Fans", "Wall Fans", "Exhaust Fans", "Pedestral Fans","Room heaters",
"Water Heaters","Smart AC's","Side-By-Side Refrigerators","Double/Triple Door Refrigerators",
 "Front Load Washing Machines","Fully Automatic Top Load Washing Machines", "Semi Automatic Washing Machines",
"Washers and Dryers","Solar Battery & Chargers","Single Door Refrigerators",
"Motion Detectors","XBOX","Playstations","Treadmills","Electronic Home Safes"]

for i in range(0,len(L1)):
    dataframe.replace(to_replace =L1[i], 
				value = L2[i], 
				inplace = True)



# writing the dataframe to another csv file 
dataframe.to_csv('outputfile.csv', 
				index = False)

