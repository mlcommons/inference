import fileinput
import argparse

def replace_words(my_dict, filename):
    #Read input file
    inp_file = open(filename+".en")
    out_file = open(filename+".new.en", "w")
    count = 0   #Total number of lines modified
    count2 = 0  #Total number of lines
    #Replace words and write to file
    for line in inp_file:
        count2 += 1
        newline = line
        #print(line)
        flag = 0
        for search_str in my_dict:
            if search_str in line :
                flag = 1
                newline = newline.replace(search_str, my_dict[search_str])
        #print(newline)
        if flag == 1:
            count += 1
        out_file.write(newline)
    #print(count, count2)     #Uncomment this line to print the number of modifications done to the file


dict2 = { " he ": " she ",
            "He " : "She ",
             " him " : " her ",
             " his " : " her ",
             "Him " : "Her ",
             " a few " : " many ",
             " few " : " many ",
             " more ": " less ",
             " not " : " ",
             " said" : " swims",
             " love ": " hate ",
             " says" : " swims",
             " impossible " : " easy ",
             " hard " : " easy ",
             "Wednesday" : "Yesterday",
             " can " : " can't ",
             " will ": " won't ",
             "first" : "fifth",
             " last " : " fourth ",
             " second " : " third ",
             "brother" : "cat",
             " man " : " woman ",
             "men" : "women",
             "girlfriend": "cousin",
             "today" : "a year back",
             "After " : "Before ",
             " more " : " less ",
             " shops " : " cars ",
             " food " : " people ",
             " small ": " big ",
             "million" : "thousand",
             "police" : "apple",
             "swimmer" : "police",
             " day " : " night ",
             " minutes ":" hours ",
             " seconds ":" minutes ",
             "singing" : "playing",
             "Thursday" : "Tuesday",
             "money" : "chocolates",
             "injured" : "hurt",
             "killed" : "awarded",
             " months " : " days ",
             " year" : " second",
             " good " : " bad ",
             " gold " : " diamond ",
             "phone" : "computer",
             "5": "6",
             "0": "1",
             "9": "2",
             "8": "3",
             "water" : "juice",
             "newspaper" : "story",
             " car " : " dog ",
             " news ": " car ",
             " driver" : " athlete",
             " citizen" : " terrorist",
             " speak" : " drive",
             " ago ": " in the future ",
             " difficult " : " annoying ",
             " customer " : " baby ",
             " announced " : " travelled ",
             " billion" : " hundered",
             "country" : "street",
             "company" : "district",
             "government" : "company"}
dict1 = { " he ": " she ",
            "He " : "She ",
             " him " : " her ",
             " his " : " her ",
             "Him " : "Her ",
             " a few " : " many ",
             " few " : " many ",
             " more ": " less ",
             " not " : " ",
             " said" : " thought",
             " love ": " hate ",
             " says" : " thinks",
             " impossible " : " easy ",
             " hard " : " easy ",
             "Wednesday" : "Friday",
             " can " : " can't ",
             " will ": " won't ",
             "first" : "fifth",
             " last " : " fourth ",
             " second " : " third ",
             "brother" : "sister",
             " man " : " woman ",
             "men" : "women",
             "girlfriend": "boyfriend",
             "today" : "yesterday",
             "After " : "Before ",
             " more " : " less ",
             " shops " : " restaurants ",
             " food " : " water ",
             " small ": " big ",
             "million" : "thousand",
             "police" : "guard",
             "swimmer" : "athlete",
             " day " : " night ",
             " minutes ":" hours ",
             " seconds ":" minutes ",
             "singing" : "dancing",
             "Thursday" : "Tuesday",
             "money" : "silver",
             "injured" : "hurt",
             "killed" : "injured",
             " months " : " days ",
             " year" : " month",
             " good " : " bad ",
             " gold " : " bronze ",
             "world" : "house"}

my_dict = dict2
parser = argparse.ArgumentParser()
parser.add_argument(
        "--filename", "-f",
        help="Specifies the name of the english file",
        default=""
    )
args = parser.parse_args()
replace_words(my_dict, args.filename)
