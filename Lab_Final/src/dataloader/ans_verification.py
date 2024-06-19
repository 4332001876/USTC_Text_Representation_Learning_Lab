import re

def verify_ans(response, label):
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


if __name__ == "__main__":
    print(verify_ans("it is pospositive", 1))
