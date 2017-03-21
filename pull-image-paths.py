import sys
import getopt
import glob
import pandas as pd

# print-out help / instructions
def help():
    print('\nusage:')
    print('python pull-image-paths.py --path=/path/to/image/directory')
    print('use --path= for setting the path to the directory with satellite images.')
    print('(note: a generic file called "image-index-summary.csv" will be written to the current folder.)\n')

# run main
def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h", ["path="])
    except getopt.GetoptError:
        sys.exit(2)

    optlist = [opt for opt, arg in opts]
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            help()
            sys.exit(2)
        elif opt == '--path':
            path = arg

    # retrieve file paths and luminosity
    files = glob.glob('%s/*/*_*.png' % path)
    df_files = pd.DataFrame(files, columns=['path'])
    df = pd.merge(df_files, df_files.path.str.split(pat='/', expand=True)[[4,5]], left_index=True, right_index=True)
    df.rename(columns={4:'intensity', 5:'filename'}, inplace=True)
    indices = pd.DataFrame(df.filename.str.replace('.png','').str.split(pat='_', expand=True)[[0,1]]).astype(int)
    indices.rename(columns={0:'i', 1:'j'}, inplace=True)
    indices['ct'] = 1
    summary = indices.set_index(['i','j']).unstack('i').stack().reset_index().groupby('j').sum().reset_index().sort(['ct','j','i'], ascending=[True, False, True])
    summary.to_csv('image-index-summary.csv')

if __name__ == "__main__":
    main(sys.argv[1:])
