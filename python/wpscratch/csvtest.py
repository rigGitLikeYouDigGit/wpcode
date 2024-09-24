
from __future__ import annotations
import typing as T
import csv, pprint
from pathlib import Path
import pandas



starPath = Path("F:\mass_downloads/5 big O/stars.csv")
planetPath = Path("F:\mass_downloads/5 big O/planets.csv")


#stars = csv.reader(starPath)


def loadCSVToDicts(csvPath, columnNames=None,
                   columnNamesFromFirstRow=True):
	"""loads the csv into dicts
	if columnNamesFromFirst, the entries of the first row determine
	names of columns

	"""
	rowDicts = []
	columnNames = columnNames or []
	with open(csvPath, encoding="utf-8") as f:
		csvReader = csv.DictReader(f,
		                           )

		for i, row in enumerate(csvReader):
			if i == 0:

				if columnNamesFromFirstRow:
					columnNames = row

				assert columnNames # check by now that we know names
				print(f'Column names are {", ".join(columnNames)}')
				continue
			csvDict = {name: row.get(name,
			                         row.get(name.title()))
			           for name in columnNames}
			rowDicts.append(csvDict)

			continue
	return rowDicts

"""
@transform(leaf=1)
def restoreNumericValues()
"""

def restoreNumericValues(dictList):
	for i in dictList:
		for k, v in dict(i).items():
			try:
				i[k] = float(v)
			except: continue
	return dictList

starDicts = loadCSVToDicts(starPath)
starDicts = restoreNumericValues(starDicts)
#for i in starDicts: print(i)
indexKey = "Seed"

# reshape into dict of { seed : [ list of { star dicts } ] }
indexedDictLists = {}
for i in starDicts:
	seed = i[indexKey]
	if not seed in indexedDictLists:
		indexedDictLists[seed] = []
	i.pop(indexKey)
	indexedDictLists[seed].append(i)


sumKey = "Unipolar Magnet (Avg)"
for seed, dicts in indexedDictLists.items():
	uniSum = 0
	for dataDict in dicts:
		uniSum += dataDict[sumKey]
	for dataDict in dicts:
		dataDict["sum"] = uniSum

seedList = list(sorted(indexedDictLists.keys(), key=lambda x: indexedDictLists[x][0]["sum"], reverse=1))
print(seedList)
32832453
