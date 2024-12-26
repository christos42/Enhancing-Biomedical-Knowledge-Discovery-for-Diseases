from Bio import Entrez
import Bio
import numpy as np
import collections
import matplotlib.pyplot as plt


class PubMed:
    def __init__(self, query, start=0):
        if len(query) == 1:
            self.query = query[0]
        else:
            self.query = ' OR '.join(query)
        self.start = start
        self.start_date, self.end_date = self.get_start_end_dates()

    def get_start_end_dates(self):
        start_dates = []
        end_dates = []
        for y in range(1900, 2023):
            for m in range(1, 12):
                start_dates.append(str(y) + '`/' + str(m))
                end_dates.append(str(y) + '/' + str(m))
                # end_dates.append(str(y) + '/' + str(m+1))

        return start_dates, end_dates

    def search(self, mindate, maxdate):
        Entrez.email = ''
        handle = Entrez.esearch(db='pubmed',
                                sort='relevance',
                                retstart=self.start,
                                retmax='10000',
                                retmode='xml',
                                datetype='pdat',
                                mindate=mindate,
                                maxdate=maxdate,
                                term=self.query)
        results = Entrez.read(handle)

        return results

    def fetch_details(self, id_list):
        id_list_c = self.check_ids(id_list)
        ids = ','.join(id_list_c)
        Entrez.email = ''
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=ids)
        results = Entrez.read(handle)

        return results

    def retrieve_abstracts(self, id_list):
        d = self.fetch_details(id_list)
        abstracts = {}
        for doc in d['PubmedArticle']:
            pmid = doc['PubmedData']['ArticleIdList'][0].split(',')[0]
            date, _ = self.get_pub_date(doc, 'article')
            try:
                #title_ = doc['MedlineCitation']['Article']['ArticleTitle']
                title = doc['MedlineCitation']['Article']['ArticleTitle']
                #if type(title_) == Entrez.Parser.StringElement:
                #    pass
                    #title = self.reform_abstract(title_)
            except:
                title = ''
            try:
                abstract = doc['MedlineCitation']['Article']['Abstract']['AbstractText']
                if type(abstract[0]) == str:
                    pass
                else:
                    abstract = self.reform_abstract(abstract)
                if type(pmid) == list:
                    abstracts[pmid[0]] = {'date': date,
                                          'title': title,
                                          'abstract': abstract}
                else:
                    abstracts[pmid] = {'date': date,
                                       'title': title,
                                       'abstract': abstract}
            except:
                pass
                # print(pmid)
        for doc in d['PubmedBookArticle']:
            pmid = doc['PubmedBookData']['ArticleIdList'][0].split(',')[0]
            date, _ = self.get_pub_date(doc, 'book_article')
            try:
                #title_ = doc['BookDocument']['ArticleTitle']
                title = doc['BookDocument']['ArticleTitle']
                #if type(title_) == Entrez.Parser.StringElement:
                #    pass
                    #title = self.reform_abstract(title_)
            except:
                title = ''
            try:
                abstract = doc['BookDocument']['Abstract']['AbstractText']
                if type(abstract[0]) == str:
                    pass
                else:
                    abstract = self.reform_abstract(abstract)
                if type(pmid) == list:
                    abstracts[pmid[0]] = {'date': date,
                                          'title': title,
                                          'abstract': abstract}
                else:
                    abstracts[pmid] = {'date': date,
                                       'title': title,
                                       'abstract': abstract}
            except:
                pass
                # print(pmid)

        return abstracts

    def total_number_of_docs(self):
        s = self.search('', '')
        print('Total number of documents: {}'.format(s['Count']))
        return s['Count']

    def retrieve_all_ids(self, print_logging=0):
        ids, n_ids_per_search = [], []
        for s_d, e_d in zip(self.start_date, self.end_date):
            s = self.search(s_d, e_d)
            ids.extend(s['IdList'])
            if print_logging:
                print('Start date: {}'.format(s_d))
                print('End data: {}'.format(e_d))
                print('{} documents found'.format(s['Count']))
                print('##########################')
            n_ids_per_search.append(s['Count'])

        unique_ids = list(set(ids))
        unique_ids_c = self.check_ids(unique_ids)

        return unique_ids_c, n_ids_per_search

    def fetch_details_all_ids(self):
        ids, _ = self.retrieve_all_ids()
        res = self.fetch_details(ids)

        return res

    def retrieve_all_abstracts(self):
        ids, _ = self.retrieve_all_ids()
        abstracts = {}
        for i in range(0, len(ids), 5000):
            if i + 5000 >= len(ids):
                d = self.fetch_details(ids[i:])
            else:
                d = self.fetch_details(ids[i:i + 5000])
            for doc in d['PubmedArticle']:
                pmid = doc['PubmedData']['ArticleIdList'][0].split(',')[0]
                date, _ = self.get_pub_date(doc, 'article')
                try:
                    title = doc['MedlineCitation']['Article']['ArticleTitle']
                except:
                    title = ''
                try:
                    abstract = doc['MedlineCitation']['Article']['Abstract']['AbstractText']
                    f = 0
                    if type(abstract[0]) == Entrez.Parser.StringElement:
                        abstract = self.reform_abstract(abstract)
                    if type(pmid) == list:
                        abstracts[pmid[0]] = {'date': date,
                                              'title': title,
                                              'abstract': abstract}
                    else:
                        abstracts[pmid] = {'date': date,
                                           'title': title,
                                           'abstract': abstract}
                except:
                    pass
                    # print(pmid)
            for doc in d['PubmedBookArticle']:
                pmid = doc['PubmedBookData']['ArticleIdList'][0].split(',')[0]
                date, _ = self.get_pub_date(doc, 'book_article')
                try:
                    title = doc['BookDocument']['ArticleTitle']
                except:
                    title = ''
                try:
                    abstract = doc['BookDocument']['Abstract']['AbstractText']
                    if type(abstract[0]) == Entrez.Parser.StringElement:
                        abstract = self.reform_abstract(abstract)
                    if type(pmid) == list:
                        abstracts[pmid[0]] = {'date': date,
                                              'title': title,
                                              'abstract': abstract}
                    else:
                        abstracts[pmid] = {'date': date,
                                           'title': title,
                                           'abstract': abstract}
                except:
                    pass
                    # print(pmid)

        return abstracts

    def reform_abstract(self, abstract):
        reformed_abstract = []
        for doc in abstract:
            reformed_abstract.append(" ".join(doc.split()))

        return " ".join(reformed_abstract)

    def get_pub_date(self, doc, doc_type):
        if doc_type == 'article':
            try:
                # 0: pubstatus: accepted
                # 1: pubstatus: PubMed upload
                # 2: pubstatus: Medline upload
                # 3: pubstatus: Entrez
                date_info = list(doc['PubmedData']['History'][1].items())
                year = date_info[0][1]
                month = date_info[1][1]
                # day = date_info[2][1]
                # date = year + '/' + month + '/' + day
                date = year + '/' + month
                found = 1
            except:
                date = ''
                found = 0
        elif doc_type == 'book_article':
            try:
                year = doc['BookDocument']['Book']['PubDate']['Year']
                month = doc['BookDocument']['Book']['PubDate']['Month']
                date = year + '/' + month
                found = 1
            except:
                date = ''
                found = 0

        return date, found

    def check_ids(self, ids):
        c_ids = []
        for id_ in ids:
            if id_ == '':
                continue
            try:
                cast = int(id_)
                c_ids.append(id_)
            except:
                print(id_)
                pass

        return c_ids


class PubMedDivide:
    def __init__(self, query, start=0):
        if len(query) == 1:
            self.query = query[0]
        else:
            self.query = ' OR '.join(query)
        self.start = start
        self.start_date, self.end_date = self.get_start_end_dates()

    def get_start_end_dates(self):
        start_dates = []
        end_dates = []
        for y in range(1900, 2023):
            for m in range(1, 12):
                start_dates.append(str(y) + '`/' + str(m))
                end_dates.append(str(y) + '/' + str(m))
                # end_dates.append(str(y) + '/' + str(m+1))

        return start_dates, end_dates

    def search(self, mindate, maxdate):
        Entrez.email = ''
        handle = Entrez.esearch(db='pubmed',
                                sort='relevance',
                                retstart=self.start,
                                retmax='10000',
                                retmode='xml',
                                datetype='pdat',
                                mindate=mindate,
                                maxdate=maxdate,
                                term=self.query)
        results = Entrez.read(handle)

        return results

    def fetch_details(self, id_list):
        ids = ','.join(id_list)
        Entrez.email = ''
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=ids)
        results = Entrez.read(handle)

        return results

    def retrieve_abstracts(self, id_list):
        d = self.fetch_details(id_list)
        abstracts = {}
        for doc in d['PubmedArticle']:
            pmid = doc['PubmedData']['ArticleIdList'][0].split(',')[0]
            date, _ = self.get_pub_date(doc, 'article')
            try:
                title = doc['MedlineCitation']['Article']['ArticleTitle']
            except:
                title = ''
            try:
                abstract = doc['MedlineCitation']['Article']['Abstract']['AbstractText']
                if type(abstract[0]) == Entrez.Parser.StringElement:
                    abstract = self.reform_abstract(abstract)
                if type(pmid) == list:
                    abstracts[pmid[0]] = {'date': date,
                                          'title': title,
                                          'abstract': abstract}
                else:
                    abstracts[pmid] = {'date': date,
                                       'title': title,
                                       'abstract': abstract}
            except:
                pass
                # print(pmid)
        for doc in d['PubmedBookArticle']:
            pmid = doc['PubmedBookData']['ArticleIdList'][0].split(',')[0]
            date, _ = self.get_pub_date(doc, 'book_article')
            try:
                title = doc['BookDocument']['ArticleTitle']
            except:
                title = ''
            try:
                abstract = doc['BookDocument']['Abstract']['AbstractText']
                if type(abstract[0]) == Entrez.Parser.StringElement:
                    abstract = self.reform_abstract(abstract)
                if type(pmid) == list:
                    abstracts[pmid[0]] = {'date': date,
                                          'title': title,
                                          'abstract': abstract}
                else:
                    abstracts[pmid] = {'date': date,
                                       'title': title,
                                       'abstract': abstract}
            except:
                pass
                # print(pmid)

        return abstracts

    def total_number_of_docs(self):
        s = self.search('', '')
        print('Total number of documents: {}'.format(s['Count']))
        return s['Count']

    def reform_abstract(self, abstract):
        reformed_abstract = []
        for doc in abstract:
            reformed_abstract.append(" ".join(doc.split()))

        return " ".join(reformed_abstract)

    def get_pub_date(self, doc, doc_type):
        if doc_type == 'article':
            try:
                # 0: pubstatus: accepted
                # 1: pubstatus: PubMed upload
                # 2: pubstatus: Medline upload
                # 3: pubstatus: Entrez
                date_info = list(doc['PubmedData']['History'][1].items())
                year = date_info[0][1]
                month = date_info[1][1]
                # day = date_info[2][1]
                # date = year + '/' + month + '/' + day
                date = year + '/' + month
                found = 1
            except:
                date = ''
                found = 0
        elif doc_type == 'book_article':
            try:
                year = doc['BookDocument']['Book']['PubDate']['Year']
                month = doc['BookDocument']['Book']['PubDate']['Month']
                date = year + '/' + month
                found = 1
            except:
                date = ''
                found = 0

        return date, found

    def process(self):
        all_abstracts = {}
        for s_d, e_d in zip(self.start_date, self.end_date):
            s = self.search(s_d, e_d)
            if len(s['IdList']) == 0:
                continue
            else:
                abstract = self.retrieve_abstracts(s['IdList'])
                all_abstracts[s_d] = abstract

        return all_abstracts


class Abstract:
    def __init__(self, abstract_dict, disease, output_path=''):
        self.abstract_dict = abstract_dict
        self.disease = disease
        self.output_path = output_path

    def number_of_abstracts(self):
        return len(list(self.abstract_dict.keys()))

    def freq_per_month(self):
        freq = {}
        for k in self.abstract_dict:
            date = self.abstract_dict[k]['date']
            if date not in freq.keys():
                freq[date] = 1
            else:
                freq[date] += 1

        return dict(collections.OrderedDict(sorted(freq.items())))

    def freq_per_year(self):
        freq = {}
        for k in self.abstract_dict:
            date = self.abstract_dict[k]['date'].split('/')[0]
            if date not in freq.keys():
                freq[date] = 1
            else:
                freq[date] += 1

        return dict(collections.OrderedDict(sorted(freq.items())))

    def plot_bar_chart_per_year(self):
        freq = self.freq_per_year()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(list(freq.keys()), list(freq.values()))
        plt.title('Released articles per year')
        plt.xlabel('Year')
        plt.ylabel('Frequency')
        plt.xticks(rotation='vertical')
        plt.yticks()
        plt.savefig(self.output_path + self.disease + '_bar_plot_freq_articles_per_year.png', bbox_inches='tight')

    def plot_per_year(self):
        freq = self.freq_per_year()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0, 0, 1, 1])
        # Excluding 2023
        #ax.plot(list(freq.keys())[:-1], list(freq.values())[:-1])
        ax.plot(list(freq.keys()), list(freq.values()))
        plt.title('Released articles per year')
        plt.xlabel('Year')
        plt.ylabel('Frequency')
        plt.xticks(rotation='vertical')
        plt.yticks()
        plt.savefig(self.output_path + self.disease + '_freq_articles_per_year.png', bbox_inches='tight')
