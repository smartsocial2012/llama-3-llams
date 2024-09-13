import pandas as pd
import json
DESCRIPTION = """\
It transforms from instruction data of (eng, eng), (kor, kor) question answer pairs
to (eng, eng), (kor, kor), (eng, kor), (kor, eng)."""
if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--from', metavar='FROM_CSV', type=str, default='../kosimcse_ncs/data/inst_data.json')
    parser.add_argument('--to', metavar='TO_CSV', type=str, default='data/inst_data_en_ko.csv')
    
    args = parser.parse_args()

    with open(args.FROM_CSV, 'r') as f:
        a = json.load(f)

    aa = []
    for x in a:
        try:
            aa.append({'user': x['eng'][0]['content'], 'assistant': x['eng'][1]['content']})
            aa.append({'user': x['kor'][0]['content'], 'assistant': x['kor'][1]['content']})
            aa.append({'user': x['eng'][0]['content'], 'assistant': x['kor'][1]['content']})
            aa.append({'user': x['kor'][0]['content'], 'assistant': x['eng'][1]['content']})
        except:
            print(x)
            raise

    df = pd.DataFrame(aa)
    df.to_csv(args.TO_CSV)