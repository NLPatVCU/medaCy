import unittest, tempfile, shutil
from medacy.tools.converters.xml_to_brat import *


sample_text = """<?xml version="1.0"?>
<?xml-stylesheet type="text/css" href="../customize/pubmed4metabo_yeast.css"?><MedlineCitationSet>
  <MedlineCitation Owner="NLM" Status="MEDLINE">
    <PMID>8435847</PMID>
    <DateCreated>
      <Year>1993</Year>
      <Month>03</Month>
      <Day>23</Day>
    </DateCreated>
    <DateCompleted>
      <Year>1993</Year>
      <Month>03</Month>
      <Day>23</Day>
    </DateCompleted>
    <DateRevised>
      <Year>2006</Year>
      <Month>11</Month>
      <Day>15</Day>
    </DateRevised>
    <Article PubModel="Print">
      <Journal>
        <ISSN IssnType="Print">0172-8083</ISSN>
        <JournalIssue CitedMedium="Print">
          <Volume>23</Volume>
          <Issue>3</Issue>
          <PubDate>
            <Year>1993</Year>
            <Month>Mar</Month>
          </PubDate>
        </JournalIssue>
        <Title>Current genetics</Title>
        <ISOAbbreviation>Curr.
Genet.</ISOAbbreviation>
      </Journal>
      <ArticleTitle>Saccharomyces cerevisiae <ENZYME><METABOLITE>phosphoglucose</METABOLITE> isomerase</ENZYME>and<ENZYME><METABOLITE><METABOLITE>fructose</METABOLITE> bisphosphate</METABOLITE> aldolase</ENZYME> can be replaced functionally by the corresponding enzymes of Escherichia coli and Drosophila melanogaster.</ArticleTitle>
      <Pagination>
        <MedlinePgn>187-91</MedlinePgn>
      </Pagination>
      <Abstract>
        <AbstractText>Two glycolytic enzymes, <ENZYME><METABOLITE>phosphoglucose</METABOLITE> isomerase</ENZYME>and<ENZYME><METABOLITE><METABOLITE>fructose</METABOLITE>-1,6-bisphosphate</METABOLITE> aldolase</ENZYME>, of Saccharomyces cerevisiae could be replaced by their heterologous counterparts from Escherichia coli and Drosophila melanogaster.
Both heterologous enzymes, which show respectively little and no sequence homology to the corresponding yeast enzymes, fully restored wild-type properties when their genes were expressed in yeast deletion mutants.
This result does not support notions of an obligatory formation of glycolytic multi-enzyme aggregates in yeast; nor does it support possible regulatory functions of yeast <ENZYME><METABOLITE>phosphoglucose</METABOLITE> isomerase</ENZYME>.</AbstractText>
      </Abstract>
      <Affiliation>Institut für Mikrobiologie, Technische Hochschule Darmstadt, Federal Republic of Germany.</Affiliation>
      <AuthorList CompleteYN="Y">
        <Author ValidYN="Y">
          <LastName>Boles</LastName>
          <ForeName>E</ForeName>
          <Initials>E</Initials>
        </Author>
        <Author ValidYN="Y">
          <LastName>Zimmermann</LastName>
          <ForeName>F K</ForeName>
          <Initials>FK</Initials>
        </Author>
      </AuthorList>
      <Language>eng</Language>
      <PublicationTypeList>
        <PublicationType>Comparative Study</PublicationType>
        <PublicationType>Journal Article</PublicationType>
        <PublicationType>Research Support, Non-U.S. Gov't</PublicationType>
      </PublicationTypeList>
    </Article>
    <MedlineJournalInfo>
      <Country>UNITED STATES</Country>
      <MedlineTA>Curr Genet</MedlineTA>
      <NlmUniqueID>8004904</NlmUniqueID>
    </MedlineJournalInfo>
    <ChemicalList>
      <Chemical>
        <RegistryNumber>0</RegistryNumber>
        <NameOfSubstance>Multienzyme Complexes</NameOfSubstance>
      </Chemical>
      <Chemical>
        <RegistryNumber>0</RegistryNumber>
        <NameOfSubstance>Recombinant Proteins</NameOfSubstance>
      </Chemical>
      <Chemical>
        <RegistryNumber>EC 4.1.2.13</RegistryNumber>
        <NameOfSubstance>Fructose-Bisphosphate Aldolase</NameOfSubstance>
      </Chemical>
      <Chemical>
        <RegistryNumber>EC 5.3.1.9</RegistryNumber>
        <NameOfSubstance>Glucose-6-Phosphate Isomerase</NameOfSubstance>
      </Chemical>
    </ChemicalList>
    <CitationSubset>IM</CitationSubset>
    <GeneSymbolList>
      <GeneSymbol>FBA1</GeneSymbol>
      <GeneSymbol>PGI</GeneSymbol>
      <GeneSymbol>PGI1</GeneSymbol>
      <GeneSymbol>PSCE-3</GeneSymbol>
    </GeneSymbolList>
    <MeshHeadingList>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Animals</DescriptorName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Cloning, Molecular</DescriptorName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Drosophila melanogaster</DescriptorName>
        <QualifierName MajorTopicYN="Y">enzymology</QualifierName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Escherichia coli</DescriptorName>
        <QualifierName MajorTopicYN="Y">enzymology</QualifierName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Fructose-Bisphosphate Aldolase</DescriptorName>
        <QualifierName MajorTopicYN="Y">genetics</QualifierName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Genetic Complementation Test</DescriptorName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Glucose-6-Phosphate Isomerase</DescriptorName>
        <QualifierName MajorTopicYN="Y">genetics</QualifierName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Glycolysis</DescriptorName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Multienzyme Complexes</DescriptorName>
        <QualifierName MajorTopicYN="N">genetics</QualifierName>
        <QualifierName MajorTopicYN="N">metabolism</QualifierName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Recombinant Proteins</DescriptorName>
        <QualifierName MajorTopicYN="N">metabolism</QualifierName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Saccharomyces cerevisiae</DescriptorName>
        <QualifierName MajorTopicYN="Y">enzymology</QualifierName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName MajorTopicYN="N">Species Specificity</DescriptorName>
      </MeshHeading>
    </MeshHeadingList>
  </MedlineCitation>
</MedlineCitationSet>"""

sample_1_expected = """T1	enzyme 24 48	phosphoglucose isomerase
T2	metabolite 24 38	phosphoglucose
T3	enzyme 51 85	fructose-1,6-bisphosphate aldolase
T4	metabolite 51 76	fructose-1,6-bisphosphate
T5	metabolite 51 59	fructose
T6	enzyme 604 628	phosphoglucose isomerase
T7	metabolite 604 618	phosphoglucose
"""


class TestXMLToBrat(unittest.TestCase):
    """Unit tests for xml_to_brat.py"""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = tempfile.mkdtemp()

        cls.xml_file_path = os.path.join(cls.test_dir, "sample.xml")
        with open(cls.xml_file_path, "w+") as f:
            f.write(sample_text)

        cls.output_file_path = os.path.join(cls.test_dir, "output_file.txt")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def test_xml_to_brat(self):
        actual, _ = convert_xml_to_brat(self.xml_file_path)
        self.assertMultiLineEqual(actual, sample_1_expected)
