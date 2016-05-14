

from xml.dom import minidom
xmldoc = minidom.parse('out.queue.xml')
itemlist = xmldoc.getElementsByTagName('interval')
numSecs = int(len(itemlist))
print numSecs
print(itemlist[0].attributes['nVehEntered'].value)
arrivals = 0
for s in itemlist:
	# print "int(s.attributes['nVehEntered'].value) = ", int(s.attributes['nVehEntered'].value)
	arrivals = arrivals + int(s.attributes['nVehEntered'].value)
print arrivals/(1.0*numSecs)