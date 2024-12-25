def main():
  import csv

  with open(r"C:\Users\rapha\Documents\datasets\tiny_stories\TinyStoriesV2-GPT4-valid.txt") as f:
    str = f.read()

  reduced_str = str[:int(len(str) / 22)] # divide by 22 to get roughly 1mb text
  reduced_str = reduced_str.replace('\n', '')
  sentences = reduced_str.split('<|endoftext|>')

  with open(r"C:\Users\rapha\Documents\datasets\tiny_stories\pico_stories.csv", 'w') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    wr.writerow(sentences)

  if __name__ == '__main__':
    main()