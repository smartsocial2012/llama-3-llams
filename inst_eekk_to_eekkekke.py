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
    parser.add_argument('--from_csv', metavar='FROM_CSV', type=str, default='../kosimcse_ncs/data/cleaned_inst_data_4k.json')
    parser.add_argument('--to_csv', metavar='TO_CSV', type=str, default='data/inst_data_en_ko_4k.json')
    
    args = parser.parse_args()

    with open(args.from_csv, 'r') as f:
        a = json.load(f)

    aa = []
    for x in a:
        try:
            aaa = []
            for inst, user_lang, assi_lang in [
                (INST_SAME_LANG, 'eng', 'eng'),
                (INST_SAME_LANG, 'kor', 'kor'),
                (INST_TO_KOR, 'eng', 'kor'),
                (INST_TO_ENG, 'kor', 'eng'),
            ]:
                aaaa = []
                aaaa.append({
                    'from': 'instruction',
                    'value': inst
                })
                for idx in range(len(x[user_lang])):
                    target_lang = user_lang if idx % 2 == 0 else assi_lang
                    aaaa.append({
                        'from': x[target_lang][idx]['role'],
                        'value': x[target_lang][idx]['content']
                    })
                
                aaa.append(aaaa)
            aa.extend(aaa)
        except:
            print(x)
            print(len(x['eng']), len(x['kor']))

    with open(args.to_csv, 'w') as f:
        for i, x in enumerate(aa):
            f.write(json.dumps({"conversation": x}) + '\n')