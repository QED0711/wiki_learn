# DATA DICTIONARY
## 01_article_comparison_info
___

## PARENT INFORMATION

### parent_title:

> The article title of the "parent" article. This only means that this is the article from which the "See Also" links were pulled.

### parent_url:

> The url of the parent article

### comparison:

> The titles of the parent and child articles (in alphabetical order)

### parent_extract:

> The summary text of the parent article

### parent_links;

> The links **within** the parent article

### parent_linkshere: 

> The links **to** the parent article

### parent_redirects:

> The links that **redirect** to the parent article
___

## CHILD INFORMATION

### child_title:

> The article title of the "child" article. This only means that this is the article from which the "See Also" links were pulled.

### child_url:

> The url of the child article

### comparison:

> The titles of the child and child articles (in alphabetical order)

### child_extract:

> The summary text of the child article

### child_links;

> The links **within** the child article

### child_linkshere: 

> The links **to** the child article

### child_redirects:

> The links that **redirect** to the child article
___

## COMPARISON INFORMATION

### parent_direct_to_child:

> Boolean - if any of the parent_links match the child_title or any links in the child_redirects

### child_direct_to_parent:

> Boolean - if any of the child_links match the parent_title or any links in the parent_redirects

### parent_path_to_child:

> number of parent_links that are found also in the child_linkshere

### child_path_to_parent:

> number of child_links that are found also in the parent_linkshere

### shared_categories:

> number of categories shared between the parent_categories and child_categories

### shared_links:

> number of links shared between the parent and child links columns

### shared_linkshere:

> number of links shared between the parent and child linkshere columns