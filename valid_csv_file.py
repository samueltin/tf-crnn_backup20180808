from src.read_dict import dict_as_str
import argparse


def checkCSVFile(file, charslist):
    csvfile = open(file[0], 'r', encoding='utf-8')
    content = csvfile.read()
    csvfile.close()
    lines = content.split('\n')
    for line_no, line in enumerate(lines):
        # print(line)
        items=line.split('\t')
        if len(items) > 1:
            label = items[1]
            chars = label.split('{')
            for char in chars:
                if not char in charslist:
                    print ('char='+char)
        else:
            print('hello '+ str(line_no))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, type=str, help='CSV filename',
                        nargs='*', default=None)
    args = vars(parser.parse_args())

    checkCSVFile(args.get('file'), dict_as_str())
