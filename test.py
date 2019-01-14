a=[[1],[],[],[3],[4],[],[]]
b=[1,2,3,4,5,6,7]
deleted = 0
for i,x in enumerate(a):
  if not x:
    b.pop(i - deleted)
    deleted = deleted + 1
while [] in a:
  a.remove([])
    
print(a)
print(b)
