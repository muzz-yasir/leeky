"""
This script demonstrates the use of the Recital, Contextual Recital, Semantic Recital,
Source Veracity, Source Recall, and Search Engine methods.
"""

# imports
import json
from pathlib import Path

# packages
from rich.console import Console
from rich.table import Table

# leeky engine imports
from leeky.engines.bloom_engine import BloomEngine
from leeky.engines.openai_engine import OpenAIEngine
from leeky.engines.gptj_engine import GTPJEngine
from leeky.engines.gptneo_engine import GTPNeoEngine
from leeky.engines.t5_engine import T5Engine

# leeky tester imports
from leeky.methods.recital import RecitalTester
from leeky.methods.contextual_recital import ContextualRecitalTester
from leeky.methods.semantic_recital import SemanticRecitalTester, SimilarityType
from leeky.methods.source_veracity import SourceVeracityTester
from leeky.methods.source_recall import SourceRecallTester
from leeky.methods.search import SearchTester



if __name__ == "__main__":
    # initialize console with wide display
    console = Console(width=200)

    # initialize default engine, which we'll assume is OpenAI text-davinci-001 for this testing
    engine = OpenAIEngine(model="gpt-4o-mini")


    # set number of samples to generate
    num_samples = 5
    max_tokens = 15

    # setup text examples
    text_1 = """"The British political and media establishment incrementally lost its collective mind over the election of Jeremy Corbyn as leader of the country’s Labour Party, and its unraveling and implosion show no signs of receding yet. Bernie Sanders is nowhere near as radical as Corbyn;
                they are not even in the same universe. But, especially on economic issues, Sanders is a more
                fundamental, systemic critic than the oligarchical power centers are willing to tolerate, and his
                rejection of corporate dominance over politics, and corporate support for his campaigns, is
                particularly menacing. He is thus regarded as America’s version of a far-left extremist,
                threatening establishment power.
                For those who observed the unfolding of the British reaction to Corbyn’s victory, it’s been
                fascinating to watch the D.C./Democratic establishment’s reaction to Sanders’ emergence
                replicate that, reading from the same script. I personally think Clinton’s nomination is extremely
                likely, but evidence of a growing Sanders movement is unmistakable. Because of the broader
                trends driving it, this is clearly unsettling to establishment Democrats — as it should be.
                A poll last week found that Sanders has a large lead with millennial voters, including young
                women; as Rolling Stone put it: “Young female voters support Bernie Sanders by an expansive
                margin.” The New York Times yesterday trumpeted that, in New Hampshire, Sanders “has
                jumped out to a 27 percentage point lead,” which is “stunning by New Hampshire standards.”
                The Wall Street Journal yesterday, in an editorial titled “Taking Sanders Seriously,” declared it is
                “no longer impossible to imagine the 74-year-old socialist as the Democratic nominee.”
                Just as was true for Corbyn, there is a direct correlation between the strength of Sanders and the
                intensity of the bitter and ugly attacks unleashed at him by the D.C. and Democratic political and
                media establishment. There were, roughly speaking, seven stages to this establishment revolt in
                the U.K. against Corbyn, and the U.S. reaction to Sanders is closely following the same script:"""

    text_2 = """GOOGLE BUILT A prototype of a censored search engine for China that links users’ searches to their personal phone numbers, thus making it easier for the Chinese government to monitor people’s queries, The Intercept can reveal.
                The search engine, codenamed Dragonfly, was designed for Android devices, and would remove content deemed sensitive by China’s ruling Communist Party regime, such as information about political dissidents, free speech, democracy, human rights, and peaceful protest.
                Previously undisclosed details about the plan, obtained by The Intercept on Friday, show that Google compiled a censorship blacklist that included terms such as “human rights,” “student protest,” and “Nobel Prize” in Mandarin.
                Leading human rights groups have criticized Dragonfly, saying that it could result in the company “directly contributing to, or [becoming] complicit in, human rights violations.” A central concern expressed by the groups is that, beyond the censorship, user data stored by Google on the Chinese mainland could be accessible to Chinese authorities, who routinely target political activists and journalists.
                Sources familiar with the project said that prototypes of the search engine linked the search app on a user’s Android smartphone with their phone number. This means individual people’s searches could be easily tracked – and any user seeking out information banned by the government could potentially be at risk of interrogation or detention if security agencies were to obtain the search records from Google.
                The search engine would be operated as part of a “joint venture” partnership with a company based in mainland China, according to sources familiar with the project. People working for the joint venture would have the capability to update the search term blacklists, the sources said, raising new questions about whether Google executives in the U.S. would be able to maintain effective control and oversight over the censorship.
                Sources familiar with Dragonfly said that the search platform also appeared to have been tailored to replace weather and air pollution data with information provided directly by an unnamed source in Beijing. The Chinese government has a record of manipulating details about pollution in the country’s cities. One Google source said that the company had built a system, integrated as part of Dragonfly, that was “essentially hardcoded to force their [Chinese-provided] data.” The source raised concerns that the Dragonfly search system would be providing false pollution data that downplayed the amount of toxins in the air.
                Google has so far declined to publicly address concerns about the Chinese censorship plans and did not respond to a request for comment on this story. It has also refused to engage with human rights groups regarding Dragonfly, ignored reporters’ questions, and rebuffed U.S. senators.
                In the meantime, pressure on Google has only intensified. On Thursday, 16 U.S. lawmakers wrote to Google CEO Sundar Pichai expressing “serious concerns” about Dragonfly and demanding information about the company’s China plans. Meanwhile, Jack Poulson, a former Google senior research scientist, told The Intercept that he was one of about five employees to have resigned from the company due to the Dragonfly plans.
                “I view our intent to capitulate to censorship and surveillance demands in exchange for access to the Chinese market as a forfeiture of our values and governmental negotiating position across the globe,” Poulson told Google bosses in his resignation letter."""

    text_3 = """
            CIA DIRECTOR MIKE Pompeo met late last month with a former U.S. intelligence official who has become an advocate for a disputed theory that the theft of the Democratic National Committee’s emails during the 2016 presidential campaign was an inside job, rather than a hack by Russian intelligence.
            Pompeo met on October 24 with William Binney, a former National Security Agency official-turned-whistleblower who co-authored an analysis published by a group of former intelligence officials that challenges the U.S. intelligence community’s official assessment that Russian intelligence was behind last year’s theft of data from DNC computers. Binney and the other former officials argue that the DNC data was “leaked,” not hacked, “by a person with physical access” to the DNC’s computer system.
            In an interview with The Intercept, Binney said Pompeo told him that President Donald Trump had urged the CIA director to meet with Binney to discuss his assessment that the DNC data theft was an inside job. During their hour-long meeting at CIA headquarters, Pompeo said Trump told him that if Pompeo “want[ed] to know the facts, he should talk to me,” Binney said.
            A senior intelligence source confirmed that Pompeo met with Binney to discuss his analysis, and that the CIA director held the meeting at Trump’s urging. The Intercept’s account of the meeting is based on interviews with Binney, the senior intelligence source, a colleague who accompanied Binney to CIA headquarters, and others who Binney told about the meeting. A CIA spokesperson declined to comment. “As a general matter, we do not comment on the Director’s schedule,” said Dean Boyd, director of the CIA’s Office of Public Affairs.
            Binney said that Pompeo asked whether he would be willing to meet with NSA and FBI officials to further discuss his analysis of the DNC data theft. Binney agreed and said Pompeo said he would contact him when he had arranged the meetings.
            It is highly unorthodox for the CIA director to reach out to someone like Binney, a 74-year-old ex-government employee who rose to prominence as an NSA whistleblower wrongfully persecuted by the government, for help with fact-finding related to the theft of the DNC emails. It is particularly stunning that Pompeo would meet with Binney at Trump’s apparent urging, in what could be seen as an effort to discredit the U.S. intelligence community’s own assessment that an alleged Russian hack of the DNC servers was part of an effort to help Trump win the presidency.
            It is possible Trump learned about Binney and his analysis by watching Fox News, where Binney has been a frequent guest, appearing at least 10 times since September 2016. In August, Binney appeared on Tucker Carlson’s Fox show to discuss his assessment that the narrative of Russia hacking the DNC during the 2016 campaign is untrue, stating that “many people are emotionally tied to this agenda, to tie the Russians to President Trump.” Binney said he is not sure how Trump found out about his analysis.
            However the meeting came about, the fact that Pompeo was apparently willing to follow Trump’s direction and invite Binney to discuss his analysis has alarmed some current and former intelligence officials.
            “This is crazy. You’ve got all these intelligence agencies saying the Russians did the hack. To deny that is like coming out with the theory that the Japanese didn’t bomb Pearl Harbor,” said one former CIA officer.
            Binney, for his part, is happy that the meeting occurred and eager to help Pompeo and Trump get to the bottom of the DNC email theft because he believes the intelligence community has not told the truth about what happened.
            “I was willing to meet Pompeo simply because it was clear to me the intelligence community wasn’t being honest here,” Binney said, referring to their assessment of the DNC email theft. “I am quite willing to help people who need the truth to find the truth and not simply have deceptive statements from the intelligence community.”
            """

    text_4 = """THE ANALYSIS BY Binney and his colleagues aligns neatly with Trump’s frequent public skepticism of the intelligence community’s assessment that Russia intervened in the 2016 campaign to damage Hillary Clinton’s candidacy and help elect Trump. The declassified summary of a U.S. intelligence community report, based on the work of the CIA, the FBI, and the NSA and made public in early January before Trump’s inauguration, stated that “Russian President Vladimir Putin ordered an influence campaign in 2016 aimed at the US presidential election. Russia’s goals were to undermine public faith in the US democratic process, denigrate Secretary Clinton, and harm her electability and potential presidency. We further assess Putin and the Russian Government developed a clear preference for President-elect Trump.” Trump has frequently raged against the accusation that he won the presidency thanks to help from the Russians, labeling the charge “fake news.”
                “The Director stands by, and has always stood by, the January 2017 Intelligence Community Assessment,” Boyd, the CIA spokesperson, said.
                Binney’s claim that the email theft was committed by an insider at the DNC also helps fuel one of the more bizarre conspiracy theories that has gained traction on the right: that the murder of a young DNC staffer last year was somehow connected to the data theft. Binney said he mentioned the case of Seth Rich to Pompeo during their meeting.
                The meeting raises questions about Pompeo’s willingness to act as an honest broker between the intelligence community and the White House, and his apparent refusal to push back against efforts by the president to bend the intelligence process to suit his political purposes. Instead of acting as a filter between Trump and the intelligence community, Pompeo’s decision to meet with Binney raises the possibility that right-wing theories aired on Fox News and in other conservative media can now move not just from conservative pundits to Trump, but also from Trump to Pompeo and into the bloodstream of the intelligence community.
                Some senior CIA officials have grown upset that Pompeo, a former Republican representative from Kansas, has become so close to Trump that the CIA director regularly expresses skepticism about intelligence that doesn’t line up with the president’s views. Pompeo has also alienated some CIA managers by growing belligerent toward them in meetings, according to an intelligence official familiar with the matter.
                The CIA, however, strongly objected to this characterization. “The Director has been adamant that CIA officers have the time, space and resources to make sound and unbiased assessments that are delivered to policy makers without fear or favor,” Boyd said in an email to The Intercept. “As he has stated repeatedly, when we deliver our assessments to policy makers, we must do so with complete candor. He has also pushed decision making down in the organization, giving officers greater ownership of their work and making them more accountable for the outcomes. These changes are designed to make CIA more agile, aggressive and responsive.”
                Yet indications of Pompeo’s willingness to support Trump at the risk of tainting the intelligence process have occasionally broken into the open in recent months. In August, the Washington Post reported that Pompeo had taken the unusual step of having the CIA’s Counterintelligence Mission Center, which would likely play a role in any inquiries by the agency into Russian election meddling, report directly to him. That move has raised concerns within the agency that Pompeo is seeking to personally control the CIA’s efforts to investigate accusations of collusion between the Trump campaign and Russia.
                More recently, at a Washington event in October, Pompeo said that U.S. intelligence had determined that Moscow’s intervention hadn’t impacted the outcome of the election. He was quickly criticized for the comments, and the CIA had to issue a clarification saying that the intelligence assessment on Russia hadn’t been altered.
                While Pompeo seems to be actively taking Trump’s side on contentious issues like Russian collusion, Dan Coats, the former Republican senator who now serves as director of national intelligence, has been largely missing in action. Coats has been reluctant to push for an aggressive Trump-Russia investigation, according to a source familiar with the matter.
                By contrast, Coats’s predecessor, James Clapper, saw himself as the public face of the intelligence community and its spokesperson. Now out of government, Clapper, who was appointed by former President Barack Obama, has become an ardent advocate for a thorough investigation of Trump and Russia. Clapper told Politico in late October that the Russian election hacking was designed to help Trump win. “The Russians have succeeded beyond their wildest expectations,” he said.
                With Coats largely out of the picture and Pompeo actively siding with Trump, the intelligence community is effectively leaderless as it struggles to come to grips with its role in the Trump-Russia inquiry. The lack of aggressive support from the intelligence community could hamper the ongoing investigation by Special Counsel Robert Mueller into possible collusion between Trump and Russia. Eventually, that lack of support could make it more difficult for Mueller’s team to glean information from inside Russia.
                Pompeo’s meeting with Binney came just days before the first charges from Mueller’s investigation were made public on October 30. Former Trump campaign chair Paul Manafort and his business partner Rick Gates were charged in a 12-count indictment in connection with their work with a pro-Russian party in Ukraine. It was also disclosed that George Papadopoulos, a former Trump foreign policy adviser, pleaded guilty earlier in October to making a false statement to the FBI in connection with his efforts to develop Russian ties during the campaign. Through a plea agreement, Papadopoulos is now cooperating with Mueller’s investigators.
                Clearly anxious as the Mueller investigation looms over his presidency, Trump has continued to promote alternative theories that would exonerate him and his campaign. Binney’s analysis falls squarely into that category."""

    # setup recital testers
    recital_tester = RecitalTester(completion_engine=engine,
                                   max_tokens=15)
    contextual_recital_tester = ContextualRecitalTester(
        completion_engine=engine, source="The Intercept", max_tokens=max_tokens
    )
    semantic_recital_tester = SemanticRecitalTester(
        completion_engine=engine,
        similarity_method=SimilarityType.SPACY_SIMILARITY_POINTS, max_tokens=max_tokens
    )

    source_recall_tester = SourceRecallTester(completion_engine=engine, min_token_proportion=0.3, max_token_proportion=0.6)

    # TODO: util function for parsing / saving output
    results = {
        "text_1": {
            "recital": recital_tester.test(text_1, num_samples=num_samples),
            "contextual_recital": contextual_recital_tester.test(
                text_1, num_samples=num_samples
            ),
            "semantic_recital": semantic_recital_tester.test(
                text_1, num_samples=num_samples
            ),
            "source_recall": source_recall_tester.test(
                text_1, match_list=["Intercept"], num_samples=num_samples
            ),
        },
        "text_2": {
            "recital": recital_tester.test(text_2, num_samples=num_samples),
            "contextual_recital": contextual_recital_tester.test(
                text_2, num_samples=num_samples
            ),
            "semantic_recital": semantic_recital_tester.test(
                text_2, num_samples=num_samples
            ),
            "source_recall": source_recall_tester.test(
                text_2, match_list=["Intercept"], num_samples=num_samples
            ),
        },
    }

    # add results for text_3 and text_4
    results.update(
        {
            "text_3": {
                "recital": recital_tester.test(text_3, num_samples=num_samples),
                "contextual_recital": contextual_recital_tester.test(
                    text_3, num_samples=num_samples
                ),
                "semantic_recital": semantic_recital_tester.test(
                    text_3, num_samples=num_samples
                ),
                "source_recall": source_recall_tester.test(
                    text_3, match_list=["Intercept"], num_samples=num_samples
                ),
            },
            "text_4": {
                "recital": recital_tester.test(text_4, num_samples=num_samples),
                "contextual_recital": contextual_recital_tester.test(
                    text_4, num_samples=num_samples
                ),
                "semantic_recital": semantic_recital_tester.test(
                    text_4, num_samples=num_samples
                ),
                "source_recall": source_recall_tester.test(
                    text_4, match_list=["Intercept"], num_samples=num_samples
                ),
            },
        }
    )

    # get the score for each tester for each text
    scores = {
        "text_1": {
            "recital": results["text_1"]["recital"]["score"],
            "contextual_recital": results["text_1"]["contextual_recital"]["score"],
            "semantic_recital": results["text_1"]["semantic_recital"]["score"],
            "source_recall": results["text_1"]["source_recall"]["score"],
        },
        "text_2": {
            "recital": results["text_2"]["recital"]["score"],
            "contextual_recital": results["text_2"]["contextual_recital"]["score"],
            "semantic_recital": results["text_2"]["semantic_recital"]["score"],
            "source_recall": results["text_2"]["source_recall"]["score"],
        },
        "text_3": {
            "recital": results["text_3"]["recital"]["score"],
            "contextual_recital": results["text_3"]["contextual_recital"]["score"],
            "semantic_recital": results["text_3"]["semantic_recital"]["score"],
            "source_recall": results["text_3"]["source_recall"]["score"],
        },
        "text_4": {
            "recital": results["text_4"]["recital"]["score"],
            "contextual_recital": results["text_4"]["contextual_recital"]["score"],
            "semantic_recital": results["text_4"]["semantic_recital"]["score"],
            "source_recall": results["text_4"]["source_recall"]["score"],
        },
    }

    # display it in a rich table with lines between rows
    table = Table(
        title="Recital Scores", show_lines=True, header_style="bold magenta", width=200
    )

    table.add_column("Text", justify="center", style="cyan")

    # show the scores from each testing method
    table.add_column("Recital", justify="center")
    table.add_column("Contextual Recital", justify="center")
    table.add_column("Semantic Recital", justify="center")
    table.add_column("Source Veracity", justify="center")
    table.add_column("Source Recall", justify="center")

    # add the rows
    table.add_row(
        "Text 1",
        str(scores["text_1"]["recital"]),
        str(scores["text_1"]["contextual_recital"]),
        str(scores["text_1"]["semantic_recital"]),
        str(scores["text_1"]["source_veracity"]),
        str(scores["text_1"]["source_recall"]),
    )
    table.add_row(
        "Text 2",
        str(scores["text_2"]["recital"]),
        str(scores["text_2"]["contextual_recital"]),
        str(scores["text_2"]["semantic_recital"]),
        str(scores["text_2"]["source_veracity"]),
        str(scores["text_2"]["source_recall"]),
    )
    table.add_row(
        "Text 3",
        str(scores["text_3"]["recital"]),
        str(scores["text_3"]["contextual_recital"]),
        str(scores["text_3"]["semantic_recital"]),
        str(scores["text_3"]["source_veracity"]),
        str(scores["text_3"]["source_recall"]),
    )
    table.add_row(
        "Text 4",
        str(scores["text_4"]["recital"]),
        str(scores["text_4"]["contextual_recital"]),
        str(scores["text_4"]["semantic_recital"]),
        str(scores["text_4"]["source_veracity"]),
        str(scores["text_4"]["source_recall"]),
    )

    # show the table
    console.print(table)

    # save the results to a json file
    with open(Path(__file__).parent / "theintercept_recital_results_small.json", "w") as f:
        json.dump(results, f, indent=4)
