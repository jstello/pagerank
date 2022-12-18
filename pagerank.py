import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    import numpy as np
    # if len(sys.argv) != 2:
    #     sys.exit("Usage: python pagerank.py corpus")
    # corpus = crawl(sys.argv[1])
    corpus = crawl('corpus0')
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    import numpy as np
    
    pages = list(corpus[page]) + list(corpus)
    weights1 = list(np.ones(len(corpus[page])) * damping_factor / len(corpus[page]))  
    weights2 = list(np.ones(len(corpus)) * (1-damping_factor)/ len(corpus)) 
    weights = weights1 + weights2
    
    prob_distribution = {page: 0 for page in pages}
    
    for i in range(len(pages)):
        prob_distribution[pages[i]] += weights[i]    
        
        
    return prob_distribution

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    import numpy as np
    pageRank = {key: 1.0  for key in corpus.keys()}
    
    error = 1
    # tol = .01
    
    # Choose a page at random
    page = random.choices(
        list(corpus.keys()),
        weights=list(pageRank.values())
        )[0]
    pageRank_previous = pageRank.copy()

    pageRank_norm = normalize_pageRank(pageRank)
    # I = 0
    error = []
    for i in range(n):
        
        # Get the transition model using d among the links in current page and 1-d among all others
        prob_dist = transition_model(corpus, page, damping_factor=damping_factor)
        
        # Choose a page according to transition model
        page = random.choices(
            list(prob_dist.keys()),
            weights=list(prob_dist.values())
        )[0]
        
        pageRank_previous = pageRank.copy()
        
        # Update the counter for the current page
        pageRank[page]+=1
        
        error.append(max_difference(normalize_pageRank(pageRank), 
                                  normalize_pageRank(pageRank_previous)))
    
    # print(f"Solution converged:")
    # print(f"Last PageRank values {normalize_pageRank(pageRank_previous)}")
    # print(f"Current PageRank values {normalize_pageRank(pageRank)}")
    # print(f"Error {error[-1]*100}%")
    
    return normalize_pageRank(pageRank)


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
   
        
        
    pageRank = {}    
        
        

        
        
    
    return pageRank

def normalize_pageRank(pageRank):
    return {key: pageRank[key]/sum(list(pageRank.values())) for key in pageRank.keys()}

def max_difference(dict1, dict2):
  # Initialize a maximum difference variable to 0
  max_diff = 0
  
  # Iterate over the keys in dict1
  for key in dict1:
    # Compute the difference between the values of the key in the two dictionaries
    diff = abs(dict1[key] - dict2[key])
    # If the difference is greater than the current maximum difference, update the maximum difference
    if diff > max_diff:
        max_diff = diff
  
  # Return the maximum difference
  return max_diff

if __name__ == "__main__":
    main()
