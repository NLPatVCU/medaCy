
# This sample will be used to test that the abstract is extracted from the xml;
# other test samples don't need to have the extra tags
xml_sample_1 = """<?xml version="1.0"?>
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

xml_sample_2 = """
      <Abstract>
        <AbstractText>KRE6 encodes a predicted type II membrane protein which, when disrupted, results in a slowly growing, killer toxin-resistant mutant possessing half the normal level of a structurally wild-type cell wall <METABOLITE>(1--&gt;6)-beta-glucan</METABOLITE> (T.
Roemer and H. Bussey, Proc.
Natl.
Acad.
Sci.
USA 88:11295-11299, 1991).
The mutant phenotype and structure of the KRE6 gene product, Kre6p, suggest that it may be a <ENZYME>beta-glucan synthase </ENZYME>component, implying that <METABOLITE>(1--&gt;6)-beta-glucan</METABOLITE> synthesis in Saccharomyces cerevisiae is functionally redundant.
To examine this possibility, we screened a multicopy genomic library for suppression of both the slow-growth and killer resistance phenotypes of a kre6 mutant and identified SKN1, which encodes a protein sharing 66% overall identity to Kre6p.
SKN1 suppresses kre6 null alleles in a dose-dependent manner, though disruption of the SKN1 locus has no effect on killer sensitivity, growth, or <METABOLITE>(1--&gt;6)-beta-glucan</METABOLITE> levels. skn1 kre6 double disruptants, however, showed a dramatic reduction in both <METABOLITE>(1--&gt;6)-beta-glucan</METABOLITE> levels and growth rate compared with either single disruptant.
Moreover, the residual <METABOLITE>(1--&gt;6)-beta-glucan</METABOLITE> polymer in skn1 kre6 double mutants is smaller in size and altered in structure.
Since single disruptions of these genes lead to structurally wild-type<METABOLITE>(1--&gt;6)-beta-glucan</METABOLITE> polymers, Kre6p and Skn1p appear to function independently, possibly in parallel, in <METABOLITE>(1--&gt;6)-beta-glucan</METABOLITE> biosynthesis.</AbstractText>
      </Abstract>"""

xml_sample_3 = """<Abstract>
        <AbstractText>Preparations of the <ENZYME><ENZYME><METABOLITE>trehalose-6-phosphate</METABOLITE> synthase</ENZYME>/<ENZYME>phosphatase</ENZYME></ENZYME> complex from Saccharomyces cerevisiae contain three polypeptides with molecular masses 56, 100 and 130 kDa, respectively.
Recently, we have cloned the gene for the 56-kDa subunit of this complex (TPS1) and found it to be identical with CIF1, a gene essential for growth on <METABOLITE>glucose</METABOLITE> and for the activity of <ENZYME><METABOLITE>trehalose-6-phosphate</METABOLITE> synthase</ENZYME>.
Peptide sequencing of the 100-kDa subunit of the <ENZYME><ENZYME><METABOLITE>trehalose-6-phosphate</METABOLITE> synthase</ENZYME>/<ENZYME>phosphatase</ENZYME></ENZYME> complex (TPS2) revealed one sequence to be 100% identical with the deduced amino acid sequence of the upstream region of PPH3 on the right arm of chromosome IV.
This sequence was used to clone an upstream region of PPH3 containing an open reading frame of 2685 nucleotides, predicted to encode a polypeptide of 102.8 kDa.
The N-terminal sequence, as well as three internal amino acid sequences, obtained from peptide sequencing of the 100-kDa subunit, were identical with specific regions of the deduced amino acid sequence.
Thus, the sequence cloned represents TPS2, the gene encoding the 100-kDa subunit of the <ENZYME><ENZYME><METABOLITE>trehalose-6-phosphate</METABOLITE> synthase</ENZYME>/<ENZYME>phosphatase</ENZYME></ENZYME> complex.
Interestingly, a stretch of about 500 aminoacids from the first part of TPS2 was 33% identical with the entire TPS1 sequence.
Disruption of TPS2 had no effect on <ENZYME><METABOLITE>trehalose-6-phosphate</METABOLITE> synthase</ENZYME> activity but caused complete loss of <ENZYME><METABOLITE>trehalose-6-phosphate</METABOLITE> phosphatase</ENZYME> activity, measured in vitro, and accumulation of excessive amounts of trehalose-6-phosphate instead of <METABOLITE>trehalose</METABOLITE> upon heat shock or entrance into stationary phase in vivo.
These results suggest that TPS2 codes for the structural gene of the <ENZYME><METABOLITE>trehalose-6-phosphate</METABOLITE> phosphatase</ENZYME>.
Heat shock induced an increase in <ENZYME><METABOLITE>trehalose-6-phosphate</METABOLITE> phosphatase</ENZYME> activity and this was preceded by an accumulation in TPS2 mRNA, suggesting that the <ENZYME><METABOLITE>trehalose-6-phosphate</METABOLITE> phosphatase</ENZYME> is subjected to transcriptional control under heat-shock conditions.</AbstractText>
      </Abstract>"""