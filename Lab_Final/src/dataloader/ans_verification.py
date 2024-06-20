import re

def verify_ans_old(response, label):
    # label is 0 or 1
    response = response.lower()
    pos_num = len(re.findall("positive", response))
    neg_num = len(re.findall("negative", response))
    # print(pos_num, neg_num)

    if pos_num > neg_num:
        response_type = 1
    elif neg_num > pos_num:
        response_type = 0
    else:
        response_type = -1
        return -1 # ambiguous

    if response_type == int(label):
        return 1 # correct
    else:
        return 0 # incorrect

def verify_ans(response, label):
    # label is 0 or 1
    response = response.lower()
    findall_result = re.findall("positive|negative", response)

    if len(findall_result) == 0:
        return -1 # ambiguous
    
    if findall_result[-1] == "positive":
        response_type = 1
    else:
        response_type = 0

    if response_type == int(label):
        return 1 # correct
    else:
        return 0 # incorrect

# "positive|negative"

if __name__ == "__main__":
    print(verify_ans("it is pospositive", 1))
