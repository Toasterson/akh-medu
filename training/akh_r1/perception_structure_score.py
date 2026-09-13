"""Examiner-only expected relations; independent of OCR interpretation."""
def expected(desc,control):
 if control=='missing_structure':return []
 if desc['id']=='table':return [dict(subject='row-'+str(i),predicate=predicate,object=value) for i,row in enumerate(desc['rows']) for predicate,value in zip(['asset','site','node'],row)]
 return [dict(subject='procedure',predicate='step-'+str(i),object=row[0]) for i,row in enumerate(desc['rows'])]
def score(desc,control,observed):
 return observed.get('facts')==expected(desc,control)
