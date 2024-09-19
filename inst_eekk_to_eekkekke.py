import pandas as pd
import json
DESCRIPTION = """\
It transforms from instruction data of (eng, eng), (kor, kor) question answer pairs
to (eng, eng), (kor, kor), (eng, kor), (kor, eng)."""
INST_SAME_LANG = """\
You are a assistant is named 'Arch AI' and developed by (주)스마트소셜(SmartSocial, Inc in English).
You have to do your best to help `user` who is chatting with you.
Try to answer in the language the user asked the question.
"""
INST_TO_ENG = """\
You are a assistant is named 'Arch AI' and developed by (주)스마트소셜(SmartSocial, Inc in English).
You have to do your best to help `user` who is chatting with you and answer in English.
"""
INST_TO_KOR = """\
You are a assistant is named 'Arch AI' and developed by (주)스마트소셜(SmartSocial, Inc in English).
You have to do your best to help `user` who is chatting with you and answer in 한국어.
"""
if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--from_csv', metavar='FROM_CSV', type=str, default='../kosimcse_ncs/data/inst_data.json')
    parser.add_argument('--to_csv', metavar='TO_CSV', type=str, default='data/inst_data_en_ko.csv')
    
    args = parser.parse_args()

    with open(args.from_csv, 'r') as f:
        a = json.load(f)

    aa = []
    for x in a:
        try:
            aa.append({
                'instruction': INST_SAME_LANG,
                'user': x['eng'][0]['content'], 
                'assistant': x['eng'][1]['content']
            })
            aa.append({
                'instruction': INST_SAME_LANG,
                'user': x['kor'][0]['content'], 
                'assistant': x['kor'][1]['content']
            })
            aa.append({
                'instruction': INST_TO_KOR,
                'user': x['eng'][0]['content'], 
                'assistant': x['kor'][1]['content']
            })
            aa.append({
                'instruction': INST_TO_ENG,
                'user': x['kor'][0]['content'], 
                'assistant': x['eng'][1]['content']
            })
        except:
            print(x)

    df = pd.DataFrame(aa)
    df.to_csv(args.to_csv)