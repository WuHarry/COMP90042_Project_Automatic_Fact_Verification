# Indexing
import sys, os, lucene, threading, unicodedata, re, codecs
from zipfile import ZipFile

from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory

# Searching
from org.apache.lucene.index import DirectoryReader
from org.apache.pylucene.queryparser.classic import PythonMultiFieldQueryParser
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher, BooleanClause

class Ticker(object):

    def __init__(self):
        self.tick = True

    def run(self):
        while self.tick:
            sys.stdout.write('.')
            sys.stdout.flush()
            time.sleep(1.0)

            
class IndexFiles(object):
    """Usage: python IndexFiles <doc_directory>"""

    def __init__(self, root, storeDir, doIndex=False):

        self.analyzer = StandardAnalyzer() 
        
        if not os.path.exists(storeDir):
            os.mkdir(storeDir)
        
        if doIndex:
            store = SimpleFSDirectory(Paths.get(storeDir))

            analyzer = LimitTokenCountAnalyzer(self.analyzer, 1048576)
            config = IndexWriterConfig(analyzer)
            config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
            writer = IndexWriter(store, config)

            self.indexDocs(root, writer)
            ticker = Ticker()
            print("commit index")
            threading.Thread(target=ticker.run).start()
            writer.commit()
            writer.close()
            ticker.tick = False
            print("done")
        
        directory = SimpleFSDirectory(Paths.get(storeDir))
        self.searcher = IndexSearcher(DirectoryReader.open(directory))

    def indexDocs(self, root, writer):

        t1 = FieldType()
        t1.setStored(True)
        t1.setTokenized(True)
        t1.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
        
        wikiFile = ZipFile(root, 'r')
        files = wikiFile.namelist()
        
        i = 0
        for file in files[1:]:
            i += 1
            wiki = wikiFile.open(file,'r')
            for line in wiki:
                for line in codecs.iterdecode(wiki, 'utf8'):
                    normailized = unicodedata.normalize('NFD', line).split(' ', 2)
                    if not normailized[1].isdigit(): continue
                    docname = normailized[0] + ' ' + normailized[1]
                    name = re.sub(r'[^a-zA-Z0-9]', ' ', normailized[0])
                    contents = normailized[2]
                    doc = Document()
                    doc.add(Field('docname', docname, t1))
                    doc.add(Field('name', name, t1))
                    doc.add(Field('contents', contents, t1))
                    writer.addDocument(doc)
            print('File %d done indexing' % i, file)
        
    def searchDocs(self, command):

        if command == '':
            return

        print("Searching for:", command)

        parser = PythonMultiFieldQueryParser(['name', 'contents'], self.analyzer)
        
        query = parser.parse(command, ['name', 'contents'], 
                             [BooleanClause.Occur.SHOULD, BooleanClause.Occur.SHOULD], self.analyzer)
        
        scoreDocs = self.searcher.search(query, 50).scoreDocs
        print("%s total matching documents." % len(scoreDocs))

        docName = []
        docContents = []
        
        for scoreDoc in scoreDocs:
            doc = self.searcher.doc(scoreDoc.doc)
            docName.append(doc.get("docname"))
            docContents.append(doc.get("contents"))
            
#             print('docname:', doc.get("docname"), 'name:', doc.get("name"), 'content:', doc.get("contents"))
        
        return docName, docContents
