with open('interactive_input/alignment', 'rb') as f:
    file = f.read()
file = file.splitlines()
file.insert(0, b'input_alignment_text = \\\n')
for i in range(1, len(file)-1):
    file[i] = file[i] + b'\\n\\\n'
file[1] = b'"'+ file[1]
file[-1] = file[-1] + b'"'
with open('interactive_input/alignment2', 'wb') as f:
    f.write(b''.join(file))

