# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

class VerySimpleStack:
  
  def __init__(self):
      self.contents = list()
      
  def push(self,item):
      self.contents.append(item)
  
  def pop(self):
      item = self.contents[-1]
      self.contents = self.contents[:-1]
      return item
  
  def pretty_print(self):
    i = 1
    print( "\n--------TOP---------")
    for item in (self.contents):
        if(i != 1):
            print( "--------------------" )           
        print( self.contents[-i])
        i+=1

    print(  "-----------------------")
    print(  "---------(table)-------")
    print(  "-----------------------")
    print(  "-- --             -- --")            
    print(  "-- --             -- --")            
    print(  "-- --             -- --")           
    print(  "-- --             -- --")           
    print(  "--                --"   )         
    print(  "--                --\n\n")            
    
      
if __name__ == '__main__':
    vss = VerySimpleStack()
    
    vss.push(1)
    vss.push(2)
    vss.push([2,3,4])
    vss.push(VerySimpleStack())
    vss.push(1)
    vss.pop()
    
    vss.pretty_print()