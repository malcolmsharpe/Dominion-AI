import re
f = file('rolling_mistakes.txt', 'w')
for line in file('program_output.txt'):
  m = re.search('rolling mistakes = ([^)]+)', line)
  if m:
    rm = m.group(1)
    if eval(rm):
      print >>f, rm
