
name = 'aj9'
prevname=['Unknown','Unknown','aj','Unknown','Unknown','Unknown','Unknown','Unknown']
if (name in prevname): 
    print("yes")
else:
    print("no") 


# def is_integer(n):
#     try:
#         float(n)
#     except ValueError:
#         return False
#     else:
#         return float(n).is_integer()
# prevname=['Unknown','Unknown','aj','Unknown','Unknown','Unknown','Unknown','Unknown']

# if __name__ == "__main__":
#     name='aj'
#     a= prevname.index(name)
#     print(a)
#     m = is_integer(a)
#     if m == True:
#         print("yes")

















# #  from flask_table import Table, Col

# # # Declare your table
# # class ItemTable(Table):
# #     name = Col('Name')
# #     description = Col('Description')

# # # Get some objects
# # class Item(object):
# #     def __init__(self, name, description):
# #         self.name = name
# #         self.description = description
# # items = [Item('Name1', 'Description1'),
# #          Item('Name2', 'Description2'),
# #          Item('Name3', 'Description3')]
# # # Or, equivalently, some dicts
# # items = [dict(name='Name1', description='Description1'),
# #          dict(name='Name2', description='Description2'),
# #          dict(name='Name3', description='Description3')]


# # def mytable():
# #     seasattendance = mongo.db.seasattendance
    
# #     emplname=[]
# #     empltemp=[]
# #     empldate_time=[]
    
# #     for names in seasattendance.find():
# #         emplname.append([names['name']])
# #         empltemp.append([names['temperature']])
# #         empldate_time.append([names['date_time']])

# #     seas = []

    
    








# # # Or, more likely, load items from your database with something like
# # items = ItemModel.query.all()

# # # Populate the table
# # table = ItemTable(items)

# # # Print the html
# # print(table.__html__())
# # # or just {{ table }} from within a Jinja template












# # pred = "('Ayush', (219, 304, 374, 150))"
# # K = ''
  
# # # using list comprehension + split() 
# # # K Character Split String 
# # res = [i for j in pred[2:].split(K) for i in (j, K)][:-1]
# # print(str(res))