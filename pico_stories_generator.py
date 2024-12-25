def main():
  import csv

  with open(r"C:\Users\rapha\Documents\datasets\tiny_stories\TinyStoriesV2-GPT4-valid.txt", 'r', encoding='utf-8') as f:
    str = f.read()

  reduced_str = str[:int(len(str) / 22)] # divide by 22 to get roughly 1mb text
  reduced_str = reduced_str.replace('\n', '')

  with open(r"data/pico_stories.txt", 'w', encoding='utf-8') as f:
    f.write(reduced_str)

if __name__ == '__main__':
    main()