"""PubMed search and abstract retrieval through the NCBI Entrez API."""

from __future__ import annotations

import collections
from collections.abc import Sequence
from datetime import date
from typing import Any

import matplotlib.pyplot as plt
from Bio import Entrez


class PubMed:
    """Search PubMed and retrieve abstracts.

    Args:
        query: The query terms, combined with OR.
        start: The index of the first search result to return.
    """

    def __init__(self, query: Sequence[str], start: int = 0) -> None:
        if len(query) == 1:
            self.query = query[0]
        else:
            self.query = " OR ".join(query)
        self.start = start
        self.start_date, self.end_date = self.get_start_end_dates()

    def get_start_end_dates(self) -> tuple[list[str], list[str]]:
        """Return the monthly date windows used to split large searches.

        There is one window per month, from January 1900 to December of the current
        year.
        """
        start_dates = []
        end_dates = []
        for y in range(1900, date.today().year + 1):
            for m in range(1, 13):
                start_dates.append(str(y) + "/" + str(m))
                end_dates.append(str(y) + "/" + str(m))
                # end_dates.append(str(y) + '/' + str(m+1))

        return start_dates, end_dates

    def search(self, mindate: str, maxdate: str) -> Any:
        """Search PubMed by publication date (up to 10,000 results, by relevance).

        Args:
            mindate: The start of the date range (``YYYY/M``), or empty for no limit.
            maxdate: The end of the date range, or empty for no limit.

        Returns:
            The Entrez search result (``IdList``, ``Count``, ...).
        """
        Entrez.email = ""
        handle = Entrez.esearch(
            db="pubmed",
            sort="relevance",
            retstart=self.start,
            retmax="10000",
            retmode="xml",
            datetype="pdat",
            mindate=mindate,
            maxdate=maxdate,
            term=self.query,
        )
        results = Entrez.read(handle)

        return results

    def fetch_details(self, id_list: list[str]) -> Any:
        """Fetch the PubMed records of the given PMIDs."""
        id_list_c = self.check_ids(id_list)
        ids = ",".join(id_list_c)
        Entrez.email = ""
        handle = Entrez.efetch(db="pubmed", retmode="xml", id=ids)
        results = Entrez.read(handle)

        return results

    def retrieve_abstracts(self, id_list: list[str]) -> dict[str, dict[str, Any]]:
        """Return the date, title and abstract of the given PMIDs.

        Articles without an abstract are skipped.

        Returns:
            PMID -> ``{"date", "title", "abstract"}``.
        """
        d = self.fetch_details(id_list)
        abstracts = {}
        for doc in d["PubmedArticle"]:
            pmid = doc["PubmedData"]["ArticleIdList"][0].split(",")[0]
            date, _ = self.get_pub_date(doc, "article")
            try:
                # title_ = doc['MedlineCitation']['Article']['ArticleTitle']
                title = doc["MedlineCitation"]["Article"]["ArticleTitle"]
                # if type(title_) is Entrez.Parser.StringElement:
                #    pass
                # title = self.reform_abstract(title_)
            except Exception:
                title = ""
            try:
                abstract = doc["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
                if type(abstract[0]) is str:
                    pass
                else:
                    abstract = self.reform_abstract(abstract)
                if type(pmid) is list:
                    abstracts[pmid[0]] = {
                        "date": date,
                        "title": title,
                        "abstract": abstract,
                    }
                else:
                    abstracts[pmid] = {
                        "date": date,
                        "title": title,
                        "abstract": abstract,
                    }
            except Exception:
                pass
                # print(pmid)
        for doc in d["PubmedBookArticle"]:
            pmid = doc["PubmedBookData"]["ArticleIdList"][0].split(",")[0]
            date, _ = self.get_pub_date(doc, "book_article")
            try:
                # title_ = doc['BookDocument']['ArticleTitle']
                title = doc["BookDocument"]["ArticleTitle"]
                # if type(title_) is Entrez.Parser.StringElement:
                #    pass
                # title = self.reform_abstract(title_)
            except Exception:
                title = ""
            try:
                abstract = doc["BookDocument"]["Abstract"]["AbstractText"]
                if type(abstract[0]) is str:
                    pass
                else:
                    abstract = self.reform_abstract(abstract)
                if type(pmid) is list:
                    abstracts[pmid[0]] = {
                        "date": date,
                        "title": title,
                        "abstract": abstract,
                    }
                else:
                    abstracts[pmid] = {
                        "date": date,
                        "title": title,
                        "abstract": abstract,
                    }
            except Exception:
                pass
                # print(pmid)

        return abstracts

    def total_number_of_docs(self) -> str:
        """Print and return the number of search results.

        The number is a string, as Entrez returns it.
        """
        s = self.search("", "")
        print("Total number of documents: {}".format(s["Count"]))
        return s["Count"]

    def retrieve_all_ids(self, print_logging: int = 0) -> tuple[list[str], list[str]]:
        """Search every monthly date window and return the unique PMIDs.

        Args:
            print_logging: If 1, print the number of results of each window.

        Returns:
            The unique numeric PMIDs, and the number of results of each window.
        """
        ids, n_ids_per_search = [], []
        for s_d, e_d in zip(self.start_date, self.end_date):
            s = self.search(s_d, e_d)
            ids.extend(s["IdList"])
            if print_logging:
                print(f"Start date: {s_d}")
                print(f"End data: {e_d}")
                print("{} documents found".format(s["Count"]))
                print("##########################")
            n_ids_per_search.append(s["Count"])

        unique_ids = list(set(ids))
        unique_ids_c = self.check_ids(unique_ids)

        return unique_ids_c, n_ids_per_search

    def fetch_details_all_ids(self) -> Any:
        """Fetch the PubMed records of all the PMIDs of the query."""
        ids, _ = self.retrieve_all_ids()
        res = self.fetch_details(ids)

        return res

    def retrieve_all_abstracts(self) -> dict[str, dict[str, Any]]:
        """Return the date, title and abstract of all the articles of the query."""
        ids, _ = self.retrieve_all_ids()
        abstracts = {}
        for i in range(0, len(ids), 5000):
            if i + 5000 >= len(ids):
                d = self.fetch_details(ids[i:])
            else:
                d = self.fetch_details(ids[i : i + 5000])
            for doc in d["PubmedArticle"]:
                pmid = doc["PubmedData"]["ArticleIdList"][0].split(",")[0]
                date, _ = self.get_pub_date(doc, "article")
                try:
                    title = doc["MedlineCitation"]["Article"]["ArticleTitle"]
                except Exception:
                    title = ""
                try:
                    abstract = doc["MedlineCitation"]["Article"]["Abstract"][
                        "AbstractText"
                    ]
                    if type(abstract[0]) is Entrez.Parser.StringElement:
                        abstract = self.reform_abstract(abstract)
                    if type(pmid) is list:
                        abstracts[pmid[0]] = {
                            "date": date,
                            "title": title,
                            "abstract": abstract,
                        }
                    else:
                        abstracts[pmid] = {
                            "date": date,
                            "title": title,
                            "abstract": abstract,
                        }
                except Exception:
                    pass
                    # print(pmid)
            for doc in d["PubmedBookArticle"]:
                pmid = doc["PubmedBookData"]["ArticleIdList"][0].split(",")[0]
                date, _ = self.get_pub_date(doc, "book_article")
                try:
                    title = doc["BookDocument"]["ArticleTitle"]
                except Exception:
                    title = ""
                try:
                    abstract = doc["BookDocument"]["Abstract"]["AbstractText"]
                    if type(abstract[0]) is Entrez.Parser.StringElement:
                        abstract = self.reform_abstract(abstract)
                    if type(pmid) is list:
                        abstracts[pmid[0]] = {
                            "date": date,
                            "title": title,
                            "abstract": abstract,
                        }
                    else:
                        abstracts[pmid] = {
                            "date": date,
                            "title": title,
                            "abstract": abstract,
                        }
                except Exception:
                    pass
                    # print(pmid)

        return abstracts

    def reform_abstract(self, abstract: list[str]) -> str:
        """Join the sections of an abstract into one string, normalizing whitespace."""
        reformed_abstract = []
        for doc in abstract:
            reformed_abstract.append(" ".join(doc.split()))

        return " ".join(reformed_abstract)

    def get_pub_date(self, doc: Any, doc_type: str) -> tuple[str, int]:
        """Return the ``YYYY/M`` date of a PubMed record and whether it was found.

        For journal articles, the second date of the record's history is used (its
        PubMed upload); for book articles, the book's publication date.

        Args:
            doc: The PubMed record.
            doc_type: ``article`` or ``book_article``.

        Returns:
            The date (empty if not found), and 1 if it was found, else 0.
        """
        if doc_type == "article":
            try:
                # 0: pubstatus: accepted
                # 1: pubstatus: PubMed upload
                # 2: pubstatus: Medline upload
                # 3: pubstatus: Entrez
                date_info = list(doc["PubmedData"]["History"][1].items())
                year = date_info[0][1]
                month = date_info[1][1]
                # day = date_info[2][1]
                # date = year + '/' + month + '/' + day
                date = year + "/" + month
                found = 1
            except Exception:
                date = ""
                found = 0
        elif doc_type == "book_article":
            try:
                year = doc["BookDocument"]["Book"]["PubDate"]["Year"]
                month = doc["BookDocument"]["Book"]["PubDate"]["Month"]
                date = year + "/" + month
                found = 1
            except Exception:
                date = ""
                found = 0

        return date, found

    def check_ids(self, ids: list[str]) -> list[str]:
        """Return the ids that are numeric, printing the others."""
        c_ids = []
        for id_ in ids:
            if id_ == "":
                continue
            try:
                int(id_)  # raises ValueError for non-numeric IDs
                c_ids.append(id_)
            except Exception:
                print(id_)
                pass

        return c_ids


class PubMedDivide:
    """Search PubMed one monthly date window at a time, with the abstracts of each.

    Args:
        query: The query terms, combined with OR.
        start: The index of the first search result to return.
    """

    def __init__(self, query: Sequence[str], start: int = 0) -> None:
        if len(query) == 1:
            self.query = query[0]
        else:
            self.query = " OR ".join(query)
        self.start = start
        self.start_date, self.end_date = self.get_start_end_dates()

    def get_start_end_dates(self) -> tuple[list[str], list[str]]:
        """Return the monthly date windows used to split large searches.

        There is one window per month, from January 1900 to December of the current
        year.
        """
        start_dates = []
        end_dates = []
        for y in range(1900, date.today().year + 1):
            for m in range(1, 13):
                start_dates.append(str(y) + "/" + str(m))
                end_dates.append(str(y) + "/" + str(m))
                # end_dates.append(str(y) + '/' + str(m+1))

        return start_dates, end_dates

    def search(self, mindate: str, maxdate: str) -> Any:
        """Search PubMed by publication date (up to 10,000 results, by relevance).

        Args:
            mindate: The start of the date range (``YYYY/M``), or empty for no limit.
            maxdate: The end of the date range, or empty for no limit.

        Returns:
            The Entrez search result (``IdList``, ``Count``, ...).
        """
        Entrez.email = ""
        handle = Entrez.esearch(
            db="pubmed",
            sort="relevance",
            retstart=self.start,
            retmax="10000",
            retmode="xml",
            datetype="pdat",
            mindate=mindate,
            maxdate=maxdate,
            term=self.query,
        )
        results = Entrez.read(handle)

        return results

    def fetch_details(self, id_list: list[str]) -> Any:
        """Fetch the PubMed records of the given PMIDs."""
        ids = ",".join(id_list)
        Entrez.email = ""
        handle = Entrez.efetch(db="pubmed", retmode="xml", id=ids)
        results = Entrez.read(handle)

        return results

    def retrieve_abstracts(self, id_list: list[str]) -> dict[str, dict[str, Any]]:
        """Return the date, title and abstract of the given PMIDs.

        Articles without an abstract are skipped.

        Returns:
            PMID -> ``{"date", "title", "abstract"}``.
        """
        d = self.fetch_details(id_list)
        abstracts = {}
        for doc in d["PubmedArticle"]:
            pmid = doc["PubmedData"]["ArticleIdList"][0].split(",")[0]
            date, _ = self.get_pub_date(doc, "article")
            try:
                title = doc["MedlineCitation"]["Article"]["ArticleTitle"]
            except Exception:
                title = ""
            try:
                abstract = doc["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
                if type(abstract[0]) is Entrez.Parser.StringElement:
                    abstract = self.reform_abstract(abstract)
                if type(pmid) is list:
                    abstracts[pmid[0]] = {
                        "date": date,
                        "title": title,
                        "abstract": abstract,
                    }
                else:
                    abstracts[pmid] = {
                        "date": date,
                        "title": title,
                        "abstract": abstract,
                    }
            except Exception:
                pass
                # print(pmid)
        for doc in d["PubmedBookArticle"]:
            pmid = doc["PubmedBookData"]["ArticleIdList"][0].split(",")[0]
            date, _ = self.get_pub_date(doc, "book_article")
            try:
                title = doc["BookDocument"]["ArticleTitle"]
            except Exception:
                title = ""
            try:
                abstract = doc["BookDocument"]["Abstract"]["AbstractText"]
                if type(abstract[0]) is Entrez.Parser.StringElement:
                    abstract = self.reform_abstract(abstract)
                if type(pmid) is list:
                    abstracts[pmid[0]] = {
                        "date": date,
                        "title": title,
                        "abstract": abstract,
                    }
                else:
                    abstracts[pmid] = {
                        "date": date,
                        "title": title,
                        "abstract": abstract,
                    }
            except Exception:
                pass
                # print(pmid)

        return abstracts

    def total_number_of_docs(self) -> str:
        """Print and return the number of search results.

        The number is a string, as Entrez returns it.
        """
        s = self.search("", "")
        print("Total number of documents: {}".format(s["Count"]))
        return s["Count"]

    def reform_abstract(self, abstract: list[str]) -> str:
        """Join the sections of an abstract into one string, normalizing whitespace."""
        reformed_abstract = []
        for doc in abstract:
            reformed_abstract.append(" ".join(doc.split()))

        return " ".join(reformed_abstract)

    def get_pub_date(self, doc: Any, doc_type: str) -> tuple[str, int]:
        """Return the ``YYYY/M`` date of a PubMed record and whether it was found.

        For journal articles, the second date of the record's history is used (its
        PubMed upload); for book articles, the book's publication date.

        Args:
            doc: The PubMed record.
            doc_type: ``article`` or ``book_article``.

        Returns:
            The date (empty if not found), and 1 if it was found, else 0.
        """
        if doc_type == "article":
            try:
                # 0: pubstatus: accepted
                # 1: pubstatus: PubMed upload
                # 2: pubstatus: Medline upload
                # 3: pubstatus: Entrez
                date_info = list(doc["PubmedData"]["History"][1].items())
                year = date_info[0][1]
                month = date_info[1][1]
                # day = date_info[2][1]
                # date = year + '/' + month + '/' + day
                date = year + "/" + month
                found = 1
            except Exception:
                date = ""
                found = 0
        elif doc_type == "book_article":
            try:
                year = doc["BookDocument"]["Book"]["PubDate"]["Year"]
                month = doc["BookDocument"]["Book"]["PubDate"]["Month"]
                date = year + "/" + month
                found = 1
            except Exception:
                date = ""
                found = 0

        return date, found

    def process(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Return the abstracts of every monthly date window with results.

        Returns:
            The start date of the window -> PMID -> abstract.
        """
        all_abstracts = {}
        for s_d, e_d in zip(self.start_date, self.end_date):
            s = self.search(s_d, e_d)
            if len(s["IdList"]) == 0:
                continue
            else:
                abstract = self.retrieve_abstracts(s["IdList"])
                all_abstracts[s_d] = abstract

        return all_abstracts


class Abstract:
    """Statistics and plots of the abstracts retrieved for a disease.

    Args:
        abstract_dict: PMID -> abstract (step 2).
        disease: The name used in the file names of the plots.
        output_path: The folder for the plots.
    """

    def __init__(
        self,
        abstract_dict: dict[str, dict[str, Any]],
        disease: str,
        output_path: str = "",
    ) -> None:
        self.abstract_dict = abstract_dict
        self.disease = disease
        self.output_path = output_path

    def number_of_abstracts(self) -> int:
        """Return the number of abstracts."""
        return len(list(self.abstract_dict.keys()))

    def freq_per_month(self) -> dict[str, int]:
        """Return the number of articles per ``YYYY/M`` date, sorted by date string."""
        freq: dict[str, int] = {}
        for k in self.abstract_dict:
            date = self.abstract_dict[k]["date"]
            if date not in freq.keys():
                freq[date] = 1
            else:
                freq[date] += 1

        return dict(collections.OrderedDict(sorted(freq.items())))

    def freq_per_year(self) -> dict[str, int]:
        """Return the number of articles per year."""
        freq: dict[str, int] = {}
        for k in self.abstract_dict:
            date = self.abstract_dict[k]["date"].split("/")[0]
            if date not in freq.keys():
                freq[date] = 1
            else:
                freq[date] += 1

        return dict(collections.OrderedDict(sorted(freq.items())))

    def plot_bar_chart_per_year(self) -> None:
        """Save a bar chart of the number of articles per year."""
        freq = self.freq_per_year()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(list(freq.keys()), list(freq.values()))
        plt.title("Released articles per year")
        plt.xlabel("Year")
        plt.ylabel("Frequency")
        plt.xticks(rotation="vertical")
        plt.yticks()
        plt.savefig(
            self.output_path + self.disease + "_bar_plot_freq_articles_per_year.png",
            bbox_inches="tight",
        )

    def plot_per_year(self) -> None:
        """Save a line plot of the number of articles per year."""
        freq = self.freq_per_year()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes([0, 0, 1, 1])
        # Excluding 2023
        # ax.plot(list(freq.keys())[:-1], list(freq.values())[:-1])
        ax.plot(list(freq.keys()), list(freq.values()))
        plt.title("Released articles per year")
        plt.xlabel("Year")
        plt.ylabel("Frequency")
        plt.xticks(rotation="vertical")
        plt.yticks()
        plt.savefig(
            self.output_path + self.disease + "_freq_articles_per_year.png",
            bbox_inches="tight",
        )
