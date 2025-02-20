brainfucked = """------------------------------------------------"""

dict_ = {">" :"52", "<" :"69", "," :"1488", "-" :"kolenval", "+" :"oboyudno", "[":"dom", "]" :"chai", ".":"🤙"}
Oboyudno = []
endl_time = 0
for ch in brainfucked:
    Oboyudno.append(f" {dict_[ch]} ")
    if endl_time == 15:
        Oboyudno.append("\n")
        endl_time=0
    else:
        endl_time+=1
print(''.join(Oboyudno))