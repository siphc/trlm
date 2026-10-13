"""Word classes. Three class systems; in each, every token belongs to exactly one named class.

ATTENTION_CLASSES  64 classes, used by the attention heads' PREFER tables (which earlier tokens
                   a head looks at, given the current token). Ward clustering of each token's left and
                   right neighbours in the TRAIN split (analysis/taxonomy.py), named by hand
                   (analysis/gen_categories.py).
CONTEXT_CLASSES    71 classes of tokens after which similar things come next: the rows of
                   CLASS_FOLLOW and CLASS_RECALL (class of the current or of the attended token) and COPY.
TARGET_CLASSES     64 classes of tokens that similar contexts predict: the columns of CLASS_FOLLOW and
                   CLASS_RECALL (class of the next token). Exceptions are grouped by them.

CONTEXT and TARGET classes were found together by information-theoretic co-clustering of the TRAIN bigram
counts (analysis/cocluster.py; no model involved): the row and column classes that keep the most
information about which token comes next. Each table file in tables/ names the systems it uses.
Each class has at most 64 tokens so that per-token tables can be written one class at a time.
Classes and their members are listed most frequent first; names were given by reading the members.
"""

ATTENTION_CLASSES = {
    'LINKERS': [
        ' and', ',', ' to', ' with', ' of', ' that', ' for', ' but', ' what', ',"', ' as', ' when',
        ' about', ' how', ' if', ' because', ' or', ' until', ' after', ' where', ':', ' than',
        ' why', ' -', ' without', ';',
    ],
    'SENTENCE_STARTS': [
        '\n', ' She', ' He', ' They', ' "', ' The', '<|eos|>', ' \n', ' It', ' I', ' One', ' But',
        ' You', ' ', 'â', ' We', ' And', ' So', ' Her', ' â', ' Then', ' His', ' From', ' Every',
        ' When', ' Suddenly', ' Let', ' Do', ' As', ' Everyone', ' Can', ' That', ' In', ' After',
        ' Now', ' There', ' Maybe', ' What', ' This', ' To', ' All', ' Thank', 'ould', '<0xC3>',
    ],
    'SENTENCE_ENDS': [
        '.', '!', '."', '!"', '?"', '?', '".',
    ],
    'DETERMINERS': [
        ' the', ' a', ' her', ' his', ' their', ' some', ' your', ' my', ' each', ' many', ' its',
        ' any', ' our', ' another',
    ],
    'LINKING_VERBS': [
        ' was', "'s", ' is', ' felt', ' looked', ' did', ' thought', ' knew', ' learned', ' has',
        ' became', ' does', ' sees', ' realized', ' looks', ' wished', ' remembered', ' forgot',
        ' makes', ' thinks', ' hoped', ' feels', ' explained',
    ],
    'SUBJECT_PRONOUNS': [
        ' he', ' she', ' you', ' they', ' there', "'t", ' we', ' who', ' let', ' thank', ' please',
        'here', 'ho', ' laug', 'hy', ' ow', 'thing', ' dro', 'nna', ' ye',
    ],
    'ADVERBS': [
        ' so', ' very', ' not', ' were', ' are', ' upon', ' all', ' always', ' never', ' just',
        "'m", ' even', ' still', ' both', ' also', ' feeling', ' being', ' ever', ' been', ' am',
        ' only', ' having', "'re", ' really', ' seen', ' enough', ' finally', ' getting', ' such',
    ],
    'ONSETS_SINGLE': [
        ' p', ' c', ' m', ' b', ' s', ' t', ' f', ' sh', ' st', ' sp', ' l', ' d', ' r', ' h',
        ' g', ' w', ' ch', ' n', ' v', ' sc', ' te', ' cl', ' fl', ' pe', ' tr', ' th', ' wh',
        ' sn', ' z', ' fe', ' gr', ' gl', ' kn', ' cut', ' qu', ' sw', ' sl', ' y', ' sk', ' br',
        ' hop', ' bl', ' cr', ' tw', ' comp', ' dr', ' squ', ' coll', ' sho', ' kne', ' ey',
        ' ben',
    ],
    'PARTICLES': [
        ' up', ' back', ' too', ' around', ' out', ' away', ' together', ' again', ' home', ' down',
        ' outside', ' more', ' sorry', ' inside', ' over', ' off', ' better', ' careful', ' safe',
        ' now', ' before', ' okay', ' right', ' closer', ' here', ' anymore', ' first', ' done',
        ' wrong', ' close', ' gone', ' alone', ' faster', ' bigger', ' everywhere', ' higher',
        ' today', ' longer', ' along', ' mine', ' less',
    ],
    'OBJECT_PRONOUNS': [
        ' it', ' them', ' him', ' something', ' me', ' one', ' this', ' us', ' everyone', ' himself',
        ' anything', ' herself', ' someone', ' everything',
    ],
    'NAMES': [
        ' Lily', ' Ben', ' Tom', ' Timmy', ' Tim', ' Sam', ' Max', ' Mom', ' Anna', ' Jack',
        ' Mia', ' Sara', ' Lucy', ' Spot', ' Mommy', ' Bob', ' Sue', ' Sarah', ' Jane', ' Dad',
        ' Joe', ' John', ' Sally', ' Daisy', ' Lila', ' Benny', ' Tommy', ' Billy', ' Amy', ' Jen',
        ' Jim', ' Daddy', ' Molly', ' Jill', ' Mary', ' Grandma', ' Emma',
    ],
    'FRAGMENTS_SHORT': [
        'er', 'y', 'e', 'a', 't', 'h', 'in', 'on', 'or', 'en', 'st', 'as', 'us', 'ld', 'ome',
        'sh', 'ally', 'ee', 'red', 'omet', 'ph', 'ually', 'llow', 'avy', 'urpr', 'chool', 'hen',
        'kay', 'ecial', 'arden', '<0xE2>', 'gry', 'aughty', '<0xC5>', '+', 'ello',
    ],
    'BARE_VERBS_TRANSITIVE': [
        ' be', ' have', ' see', ' make', ' find', ' get', ' take', ' eat', ' say', ' feel', ' try',
        ' share', ' keep', ' explore', ' give', ' watch', ' use', ' pick', ' tell', ' open',
        ' fix', ' touch', ' catch', ' hold', ' think', ' read', ' climb', ' buy', ' hear', ' ask',
        ' leave', ' enjoy', ' wear', ' reach', ' pull', ' call', ' follow', ' throw', ' bring',
        ' fill',
    ],
    'FRAGMENTS_VC': [
        'ar', 'at', 'an', 'ad', 'id', 'ot', 'it', 'ay', 'el', 'ow', 'il', 'ill', 'ur', 'ic',
        'op', 'um', 'un', 'oo', 'ag', 'ut', 'ip', 'am', 'ap', 'ight', 'own', 'im', 'ood', 'ack',
        'ell', 'em', 'ound', 'iss', 'ast', 'all', 'oth', 'ass', 'amp', 'up', 'uck', 'ors', 'umb',
        'iny', 'urt', 'uff', 'ream', 'oy',
    ],
    'FRAGMENTS_ENDINGS': [
        'le', 'm', 'et', 'al', 'ch', 'k', 'ent', 'est', 've', 'ful', 'ry', 'ate', 'nd', 'ck',
        'ves', 'ge', 'ant', 'ep', 'ous', 'able', 'ble', 'ure', 'ved', 'ars', 'th', 'oon', 'ving',
        'ation', 'ven', 'ber', 'ter', 'ich', 'ious', 'icy', 'der', 'ty', 'orm', 'thy', 'less',
        'de', 'bble', 'ild', 'ible', 'cy', 'vel', 'very', 'ek', 'bow', 'mer', 'ail', 'etter',
        'ook', 'V', 'row', 'ence', 'rel', 'med', 'isy', 'sed', 'ture', 'ren', '0', '3', 'more',
    ],
    'PAST_VERBS_MOTION': [
        ' had', ' went', ' ran', ' got', ' came', ' played', ' lived', ' walked', ' flew', ' kept',
        ' stopped', ' quickly', ' fell', ' jumped', ' waved', ' hopped', ' worked', ' happily',
        ' sat', ' set', ' carefully', ' lost', ' needed', ' left', ' finished', ' arrived', ' filled',
        ' stayed', ' turned', ' climbed', ' slowly', ' broke', ' stepped', ' drove', ' sang',
        ' moved', ' seemed', ' swam',
    ],
    'PAST_VERBS_TRANSITIVE': [
        ' saw', ' asked', ' made', ' put', ' found', ' took', ' told', ' gave', ' heard', ' hugged',
        ' thanked', ' opened', ' picked', ' noticed', ' showed', ' watched', ' grabbed', ' helped',
        ' reached', ' ate', ' used', ' held', ' enjoyed', ' followed', ' met', ' brought', ' pulled',
        ' shared', ' threw', ' closed', ' pushed', ' hit', ' hugs', ' shook', ' caught', ' dropped',
    ],
    'BARE_VERBS_ACTIVITY': [
        ' play', ' go', ' help', ' do', ' come', ' hurt', ' show', ' clean', ' run', ' wait',
        ' listen', ' walk', ' stay', ' fly', ' learn', ' stop', ' jump', ' draw', ' swim', ' sleep',
        ' laugh', ' move', ' build', ' start', ' break', ' fight', ' fall', ' talk', ' sing',
        ' pet', ' sit', ' grow', ' ride', ' visit', ' hide', ' paint', ' dance', ' join', ' win',
        ' turn', ' cook', ' heal', ' Mr', ' stand', ' free', ' dry', ' happen', ' drink', ' shout',
        ' flow', ' neigh',
    ],
    'FAMILY_AND_BODY': [
        ' mom', ' friends', ' friend', ' mommy', ' dad', ' toy', ' way', ' family', ' hand',
        ' idea', ' eyes', ' hands', ' head', ' voice', ' face', ' place', ' adventure', ' brother',
        ' mum', ' thing', ' name', ' hair', ' parents', ' surprise', ' lesson', ' arm', ' mother',
        ' grandma', ' daddy', ' fur', ' tail', ' plan', ' owner', ' job', ' sister', ' nap',
        ' mouth', ' heart', ' pocket', ' teacher', ' wings', ' neck', ' secret', ' favor',
    ],
    'PREPOSITIONS': [
        ' in', ' on', ' at', ' from', ' into', ' by', ' under', ' through', ' near', ' behind',
        ' towards', ' across',
    ],
    'SUFFIXES_ING_ED': [
        'ing', 'ed', 'ive', 'led', 'king', 'ite', 'ted', 'hing', 'ish', 'ked', 'ched', 'hed',
        'hes', 'ide', 'are', 'aring', 'ered', 'ged', 'ined', 'ore', 'ain', 'ting', 'ced', 'kes',
        'ated', 'ling', 'ased', 'ire', 'ried', 'ipped', 'ming', 'ith', 'ank', 'cked', 'iting',
        'pping', 'pped', 'ning', 'ches', 'ough', 'rew', 'ized', 'ised', 'ied', 'ited', 'ull',
        'ken', 'ught', 'ened', 'ange', 'asure', 'apped', 'ired', 'bbed', 'ished', 'ued', 'ained',
        'ther', 'iced', 'ached', 'ided',
    ],
    'FRAGMENTS_MIXED': [
        'Once', 'o', ' named', '€', 'i', 'u', 'r', 'ir', 're', 'ra', 'œ', '€™', 'ri', 'ro', 'he',
        'bo', ' years', 'ul', 'ac', 'ere', 'br', 'pl', ' year', 'col', 'ret', 'sp', 'au', 'gan',
        'ittle', 'j', 'get', 'ht', 'eci', 'av', '<0x80>', '<0xE2><0x82><0xAC><0xE2>', 'ec', 'fu',
        'lc', 'ott', 'itc', '<0xCB>', 'so', '<0xC2>', 'asket',
    ],
    'SUFFIXES_PLURAL': [
        's', 'es', 'ly', 'ers', 'ice', 'les', 'om', 'ion', 'ue', 'ies', 'ace', 'ater', 'ient',
        'ons', ' else', 'room', 'ise', 'orn', ' cream', 'ucky', 'ings', 'gs', 'ope', 'fe', 'ze',
        'ins', 'eared', 'ier', 'ment', 'ely', 'ces', 'ions', 'self', 'oney', 'leep', 'way', 'cess',
        'ime', 'ouse', 'gon', 'fully', 'ises', 'me', 'set', 'reed', 'ner', 'ways', 'ident', 'ount',
        'oud',
    ],
    'SIZE_AND_COLOUR': [
        ' little', ' big', ' new', ' special', ' small', ' loud', ' long', ' beautiful', ' red',
        ' great', ' blue', ' bright', ' green', ' yellow', ' tall', ' wonderful', ' pink', ' few',
        ' white', ' deep', ' real', ' strange', ' black', ' magical', ' brown', ' magic', ' young',
        ' nearby', ' wise', ' purple', ' orange', ' blank',
    ],
    'TIME_WORDS': [
        ' day', ' time', ' then', ' soon', ' while', ' night', 'day', ' morning', ' sometimes',
        'ever', 'lease', 'side', 'where', ' suddenly', ' course', 'ventually', 'imes', ' welcome',
        ' though', 'inally', 'gether', 'bye', 'uch', 'ember',
    ],
    'CREATURES': [
        ' girl', ' boy', ' bird', ' man', ' dog', ' bear', ' cat', ' fish', ' rabbit', ' bunny',
        ' lady', ' mouse', ' flower', ' butterfly', ' frog', ' baby', ' squirrel', ' lion', ' puppy',
        ' fox', ' train', ' monster', ' princess', ' dragon', ' monkey', ' farmer', ' bee', ' doctor',
        ' fairy', ' animal', ' child', ' elephant', ' star', ' owl', ' kid',
    ],
    'MASS_AND_PLURAL_NOUNS': [
        ' fun', ' things', ' toys', ' water', ' animals', ' people', ' noise', ' bed', ' food',
        ' flowers', ' work', ' cry', ' cake', ' birds', ' kids', ' cars', ' blocks', ' trees',
        ' paper', ' books', ' children', ' colors', ' leaves', ' cookies', ' dinner', ' others',
        ' pictures', ' clothes', ' joy', ' music', ' money', ' pieces', ' candy', ' words', ' dolls',
        ' twins', ' school', ' games', ' noises', ' late', ' stories', ' shoes', ' adventures',
        ' lunch', ' apple', ' soup', ' stars', ' rocks', ' shapes', ' wood', ' butter', ' pie',
        ' danger',
    ],
    'OPENERS_UNSPACED': [
        'The', '"', 'One', 'But', 'S', 'When', 'J', 'Suddenly', 'A', 'So', 'L', 'Her', 'After',
        'As', 'E', 'Then', 'From', 'His', 'At', 'Finally', 'On', 'And', 'Jo', 'K',
    ],
    'SPEECH_VERBS': [
        ' said', ' smiled', ' says', ' laughed', ' nodded', ' replied', ' shouted', ' cried',
        ' agreed', ' smiles', ' asks', ' waited', ' clapped', ' cheered',
    ],
    'LETTERS_LOWER': [
        'p', 'l', 'g', 'b', 'c', 'f', 'n', 'd', 'll', 'is', 'v', 'z', 'pp', 'co', 'bb', 'os',
        'ol', 'ff', 'rom', 'imb',
    ],
    'FEELINGS': [
        ' happy', ' sad', ' excited', ' scared', ' much', ' angry', ' fast', ' high', ' curious',
        ' glad', ' tired', ' surprised', ' tight', ' far', ' hungry', ' amazed', ' worried',
        ' grateful', ' well', ' upset', ' lonely', 'fter',
    ],
    'VERBS_BEFORE_TO': [
        ' wanted', ' loved', ' started', ' liked', ' decided', ' tried', ' began', ' likes',
        ' continued', ' wants', ' listened', ' runs', ' promised',
    ],
    'QUALITIES': [
        ' good', ' nice', ' kind', ' hard', ' bad', ' brave', ' pretty', ' shiny', ' strong',
        ' different', ' warm', ' sweet', ' yummy', ' soft', ' funny', ' cool', ' dark', ' cold',
        ' delicious', ' friendly', ' hot', ' broken', ' scary', ' wet', ' amazing', ' perfect',
        ' silly', ' quiet', ' colorful', ' smart', ' dangerous', ' heavy', ' dirty', ' gentle',
        ' naughty', ' fair', ' slow', ' quick',
    ],
    'OBJECTS': [
        ' tree', ' house', ' ball', ' room', ' box', ' truck', ' boat', ' doll', ' book', ' story',
        ' game', ' bag', ' hole', ' wall', ' castle', ' balloon', ' field', ' party', ' pot',
        ' cup', ' blanket', ' bike', ' tower', ' bowl', ' bath', ' basket', ' chair', ' key',
        ' jar', ' bucket', ' farm', ' bench',
    ],
    'PLACES': [
        ' park', ' garden', ' sky', ' door', ' ground', ' forest', ' store', ' kitchen', ' grass',
        ' world', ' pond', ' floor', ' window', ' air', ' table', ' hill', ' beach', ' woods',
        ' lake', ' river', ' town', ' swings', ' yard', ' sea', ' street', ' fence', ' shop',
        ' backyard',
    ],
    'MODALS': [
        ' can', ' could', ' would', ' will', ' should', ' might', "'ll", ' must',
    ],
    'NAMES_UNSPACED': [
        'Lily', 'They', 'Tom', 'He', 'Mom', 'Ben', 'She', 'Anna', 'Sara', 'Mommy', 'Tim', 'Timmy',
        'John', 'Sam', 'Jack', 'Max', 'Mia', 'Lila',
    ],
    'DETERMINERS_RARE': [
        ' an', ' no', ' every', ' next', '-', ' three', ' creat', ' ju', ' pol', ' pat', ' int',
        ' con', ' gu', ' cle', ' imp', ' em', ' disapp', ' del', ' att', ' smo', ' inst', ' trou',
        ' dist', ' fore', ' relie', ' 3', ' stu', ' af', ' happ', ' rel', ' cont', ' fam', ' ab',
        ' advent', ' dif', ' delic', ' sec', ' thr', ' amaz', ' qui', ' purp', ' beauti', ' somet',
        ' ke',
    ],
    'CAPITALS_SPACED': [
        ' M', ' B', ' S', ' A', ' L', ' P', ' T', ' W', ' C', ' F', ' R', ' J', ' E', ' D', ' G',
        ' H', ' K', ' N', ' Fl', ' Jo', ' Em', ' Gra', ' Sp', ' Th',
    ],
    'ONSETS_A': [
        ' re', ' e', ' li', ' wa', ' lo', ' le', ' dri', ' dis', ' jo', ' wor', ' en', ' al',
        ' ha', ' ra', ' tra', ' mar', ' dan', ' ad', ' pre', ' bran', ' pro', ' u', ' ar', ' mat',
        ' des', ' stor', ' ac', ' per', ' dra', ' tre', ' poin', ' ex', ' gif', ' prin', ' min',
        ' sor', ' dir', ' bel', ' pret', ' squir', ' cour', ' ta', ' tal', ' accident', ' woke',
        ' gent', 'lp', ' underst', ' hun', ' vis', ' welc', ' belie', ' wat', ' teac', ' grate',
    ],
    'ING_FORMS': [
        ' playing', ' look', ' going', ' walking', ' looking', ' important', ' mean', ' ready',
        ' making', ' coming', ' doing', ' running', ' able', ' trying', ' flying', ' helping',
        ' crying', ' stuck', ' eating', ' singing', ' exploring', ' taking',
    ],
    'ONSETS_B': [
        ' co', ' o', ' che', ' un', ' ro', ' se', ' la', ' com', ' tri', ' gra', ' cra', ' bo',
        ' fa', ' fo', ' pen', ' pa', ' je', ' pu', ' plac', ' su', ' ph', ' ca', ' ba', ' bra',
        ' clos', ' din', ' cu', ' ma', ' loo', ' bro', ' nam', ' mus', ' el',
    ],
    'MENTAL_VERBS': [
        ' like', ' want', ' know', ' need', ' sure', ' worry', ' understand', ' remember', ' believe',
        ' forget',
    ],
    'QUOTE_OPENERS': [
        'I', 'It', 'You', 'Let', 'What', 'That', 'Thank', 'We', 'Can', 'Don', 'This', 'Maybe',
        'Why', 'Do',
    ],
    'ONSETS_C': [
        ' sa', ' mo', ' fr', ' ru', ' po', ' bu', ' app', ' ho', ' gi', ' mu', ' de', ' exc',
        ' spark', ' answ', ' bott', ' cho', ' ve', ' bar', ' butt', ' ste', ' pers', ' corn',
        ' exp', ' lea', ' wo', ' clo', ' sto', ' op', ' fi', ' vo', ' butterf', ' pi', ' cast',
        ' fro', ' buck', ' mon',
    ],
    'MODIFIERS': [
        ' other', ' old', ' best', ' two', ' favorite', ' own', ' teddy', ' living', ' most',
        ' same', ' neighb', ' journ', ' war', ' bir', ' pic', ' mag', ' fav', ' ele', ' stre',
        ' pare', ' differe', ' curi', ' import', ' hig', ' lon', 'fere', ' mor', ' wal',
    ],
    'THINGS_OUTDOORS': [
        ' car', ' sun', ' slide', ' rain', ' rock', ' dress', ' sand', ' fire', ' swing', ' ice',
        ' wind', ' snow', ' race', ' mail', ' color', ' block',
    ],
    'FRAGMENTS_D': [
        'se', 'ce', 'ny', 'ke', 'ess', 'zy', 'lf', 'ens', 'dy', 'ey', 'man', 'ary', 'cle', 'ear',
        'be', 'bby', 'ppy', 'by', 'ger', 'lly', 'unny', 'rot', 'key', 'cket', 'bbit', 'phant',
    ],
    'ABSTRACT_NOUNS': [
        ' proud', ' end', ' lots', ' lot', ' care', ' rest', ' top', ' full', ' picture', 'ance',
        ' moral', ' side', 'ts', ' piece', 'ead', ' part', 'oup', ' list', 'eces', ' goodbye',
        ' happened', ' yes', ' turns', 'fused', 'ward', 'llo', 'ds', 'yard',
    ],
    'VERB_ONSETS': [
        ' k', ' pretend', ' tast', ' smell', ' dream', ' hid', ' str', ' count', ' wonder', ' sail',
        ' roll', ' scream', ' pass', ' spr', ' mix', ' pack', ' miss', ' belong', ' gre', ' yell',
        'ng', ' smil', 'app', 'joy', ' shap', 'maz',
    ],
    'THINGS_MISC': [
        ' sound', ' stick', ' hat', ' duck', ' light', ' spot', ' bug', ' song', ' treat', ' land',
        ' cloud', ' pain', ' snack', ' plant', ' treasure', ' sign', ' carrot', ' fruit', ' cow',
        ' bow',
    ],
    'NAME_ENDINGS': [
        'ie', 'my', 'ane', 'ama', 'ummy', 'eter', 'ob', 'uffy', 'ily', 'ia', 'olly', 'illy',
        'aughter', 'andy', 'eddy', 'ike', 'ucy', 'ma', 'ara', 'aisy', 'ames', 'ndma', 'ila',
    ],
    'FRAGMENTS_G': [
        'w', 'ig', ' called', 'iz', 'if', 'udd', 'ang', 'qu', 'zz', 'ugg', 'igg', 'ong', 'riend',
        'urp', '/', 'imal', 'sw', 'hn', 'aught', 'q',
    ],
    'CAPITALS_UNSPACED': [
        'M', 'B', 'W', 'O', 'G', 'D', 'P', 'H', 'C', 'T', 'N', 'R', 'F', 'Y', 'Z', 'U', 'Q',
        'Th',
    ],
    'FRAGMENTS_H': [
        'ver', 'ust', 'ect', 'ather', 'ss', 'end', 'oom', 'fort', 'ft', 'nder', 'pt', 'iver',
        'ax', 'irl', 'wn', 'vent', 'ine', 'one', 'ake', 'ase', 'ort', 'ape', 'ord', 'ade', 'ocked',
        'adow', 'akes', 'ock', 'ame', 'ield', 'age', 'ard', 'ave', 'art', 'ash', 'ries', 'ou',
        'her', 'ared', 'ush', 'ity', 'ards', 'ross', 'ause', 'hat', 'aut', 'oug', 'ist', 'itt',
        'ark', 'ach', 'ind', 'atter', 'ump', 'issed', 'ountain', 'aybe', 'ask', 'elt', 'unch',
        'ister',
    ],
    'AFFECTION_NOUNS': [
        ' love', ' hug', ' smile', ' bit', 'ct', 'nt', ' moment', ' bite', ' wish', 'dd', 'dded',
        'nts',
    ],
    'NEGATED_AUXILIARIES': [
        ' didn', ' couldn', ' don', ' wasn', ' won', ' wouldn',
    ],
    'INTERJECTIONS': [
        'Yes', 'No', 'Look', 'Wow', 'Hello', 'OK', 'Oh', 'Okay', 'Sure', 'Hi',
    ],
    'ONSETS_D': [
        ' pr', ' wr', ' spl', ' sear', ' rem', ' dec', ' im', ' pus', ' forg', ' inv', ' es',
        ' resp', ' prom', ' fin', ' cre', ' expl', ' bec', ' ag', ' perf', ' beh', ' repl', ' sm',
        ' contin', ' acc', ' unt', ' arri', ' j', ' pl', ' bre', ' cray', ' tom', ' whe', ' past',
        ' tow', ' col', ' ne', ' flo', ' pap', ' surpr', ' ide', ' clot',
    ],
    'FRAGMENTS_L': [
        'our', 'ile', 'ath', 'ress', 'ower', 'air', 'ool', 'ract', 'out', 'ick', 'ug', 'uc',
        'ink', 'itting', 'uit', 'igh', 'udden', 'ix', 'outh', 'ourn', 'and', 'urn', 'ored', 'orrow',
        'oun', 'ought', 'arn', 'ree', 'lew', 'oring', 'other', ' bus', ' mess', 'nn', 'ster',
        'eet', 'sel', 'irst', 'ctor', ' beaut', 'x', 'ak', 'ady', 'xt', 'fore', 'ople',
    ],
    'DIGITS_AND_SYMBOLS': [
        "'", 'ign', '1', '2', 'X', '5', '4', '9', '`', '8', '6', '7', ')', '(', '$', '*', '_',
        '\\', '&', '=', '<', '[', '#', '\t',
    ],
    'BYTES_AND_RARE_SYMBOLS_1': [
        '<0x9C>', '<0x9D>', '<0x83>', '<0x82>', '<0xB9>', '<0xB0>', '<0x9A>', '<0xA1>', '<0xAF>',
        '<0xA5>', '<0xA7>', '<0x9E>', '<0xA4>', '<0xA6>', '<0xA9>', '<0xA0>', '<0xAD>', '<0xB4>',
        '<0xB1>', '<0xA2>', '<0xB3>', '<0xAC>', '<0xBF>', '<0xA8>', '<0xBD>', '<0xBB>', '<0xA3>',
        '<0xBE>', '<0xB8>', '<0xAA>', '<0xBA>', '<0xAB>', '<0xBC>', '<0xB7>', '<0xB6>', '>',
        ']', '%', 'itchen', 'uddenly', '<0x84>', '<0x86>', '<0x92>', '<0x85>', '<0x8A>', '<0x89>',
        '~', '<0x99>', '<0x93>', '<0x98>', '<0x84><0xA2>', '<0xAE>', '<0xB2>', '<0xB5>', '@',
        '|', '{', '}', '<0x91>', '<0xC6>',
    ],
    'NEVER_IN_TRAINING_TEXT_1': [
        '^', '<0xC0>', '<0xC1>', '<0xC4>', '<0xC7>', '<0xC8>', '<0xC9>', '<0xCA>', '<0xCC>',
        '<0xCD>', '<0xCE>', '<0xCF>', '<0xD0>', '<0xD1>', '<0xD2>', '<0xD3>', '<0xD4>', '<0xD5>',
        '<0xD6>', '<0xD7>', '<0xD8>', '<0xD9>', '<0xDA>', '<0xDB>', '<0xDC>', '<0xDD>', '<0xDE>',
        '<0xDF>', '<0xE0>', '<0xE1>', '<0xE3>', '<0xE4>', '<0xE5>', '<0xE6>', '<0xE7>', '<0xE8>',
        '<0xE9>', '<0xEA>', '<0xEB>', '<0xEC>', '<0xED>', '<0xEE>', '<0xEF>', '<0xF0>', '<0xF1>',
        '<0xF2>', '<0xF3>', '<0xF4>', '<0xF5>', '<0xF6>', '<0xF7>', '<0xF8>', '<0xF9>', '<0xFA>',
        '<0xFB>', '<0xFC>', '<0xFD>', '<0xFE>', '<0xFF>', '\x00', '\x01', '\x02', '\x03', '\x04',
    ],
    'NEVER_IN_TRAINING_TEXT_2': [
        '\x05', '\x06', '\x07', '\x08', '\x0b', '\x0c', '\r', '\x0e', '\x0f', '\x10', '\x11',
        '\x12', '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a', '\x1b', '\x1c',
        '\x1d', '\x1e', '\x1f', '\x7f', '<0x81>', '<0x87>', '<0x88>', '<0x8B>', '<0x8C>', '<0x8D>',
        '<0x8E>', '<0x8F>', '<0x90>', '<0x94>', '<0x95>', '<0x96>', '<0x97>', '<0x9B>', '<0x9F>',
        '<0xE2><0x82>',
    ],
}

CONTEXT_CLASSES = {
    'SENTENCE_ENDS': [
        '.', '!', '".', '<0xB9>',
    ],
    'THE_AND_LITTLE': [
        ' the', ' The', ' little', 'The', ' each', ' years',
    ],
    'LINE_STARTS_AND_SYMBOLS': [
        '\n', ' "', '"', ' ', 'œ', '`', '$', '<0xA0>', '\\', '=', '<0xA2>', '<0xA1>', '#', '\t',
        '<0xAF>', '<0xA5>', '<0xBF>', '<0xA7>', '<0xBE>', '<0xAB>', '<0xBC>', '{', '<0x8A>',
        '@',
    ],
    'TO_AND_MODALS': [
        ' to', ' not', "'t", ' can', ' could', ' would', ' will', ' or', ' should', ' might',
        "'ll", ' must', 'ways', 'nna',
    ],
    'AND_AND_ADVERBS': [
        ' and', ' always', ' never', ' just', ' still', ' both', ' also', ' only', ' finally',
        'ually', '&',
    ],
    'FORMS_OF_BE': [
        ' was', ' is', ' were', ' be', ' are', ' felt', ' did', "'m", ' kept', ' feeling', ' being',
        ' been', ' became', ' am', ' does', "'re", ' getting',
    ],
    'COMMAS': [
        ',', ',"', ':', 'ittle', '<0x9C>',
    ],
    'HE_AND_SHE': [
        ' She', ' He', ' he', ' she', ' who', 'He', 'She',
    ],
    'NAMES_AND_FAMILY': [
        ' Lily', ' mom', ' Ben', ' Tom', ' Timmy', 'Lily', ' Tim', ' mommy', ' Sam', ' Max',
        ' dad', ' Mom', 'Tom', ' Anna', ' Jack', ' Mia', ' Sara', 'Mom', 'Ben', 'my', ' Lucy',
        ' Spot', 'ny', ' quickly', ' Mommy', 'Anna', ' Bob', ' Sue', 'Sara', ' Sarah', ' Jane',
        ' Dad', ' mum', 'Mommy', 'Tim', ' John', ' parents', 'Timmy', 'John', 'Sam', ' carefully',
        'ally', ' Sally', ' Daisy', ' Lila', ' Benny', 'Jack', ' Tommy', 'ama', ' mother', ' grandma',
        ' daddy', 'ummy', ' Billy', ' Amy', 'ily', 'Max', ' slowly', ' Daddy', ' Molly', 'Mia',
        ' Jill', 'olly', 'illy',
    ],
    'POSSESSIVES': [
        ' her', ' his', "'s", ' their', ' new', ' your', ' my', ' best', ' Her', ' His', ' two',
        ' its', ' our', 'Her', 'His',
    ],
    'PHRASE_ENDINGS': [
        ' fun', ' away', ' together', ' park', ' again', ' home', ' toys', ' outside', ' tree',
        ' house', ' water', ' room', ' better', ' garden', ' sky', ' safe', ' door', ' noise',
        'age', ' ground', 'les', ' idea', ' smile', ' food', ' flowers', ' forest', ' slide',
        ' okay', ' face', ' store', ' cry', ' cake', ' here', ' anymore', ' book', ' adventure',
        ' kitchen', ' happened', ' grass', ' pond', ' blocks', ' floor', ' paper', ' hole', ' wrong',
        ' books', ' table', ' hill', ' gone', 'room', ' leaves', ' cookies', ' beach', 'ries',
        ' dinner', ' castle', ' alone', ' clothes', ' tail', ' joy', ' woods', ' amazed', 'ape',
        ' job',
    ],
    'BEFORE_A_COMMA': [
        ' said', ' day', ' time', ' says', ' then', 'Yes', ' say', ' So', ' Then', 'No', ' while',
        ' Suddenly', 'Suddenly', ' replied', 'So', 'Look', ' night', ' shouted', ' done', ' worry',
        'Wow', 'Hello', ' Now', 'Then', 'OK', ' morning', 'Okay', 'Sure', ' everywhere', 'Hi',
        ' course', 'ventually', 'imes', ' welcome', 'Finally', 'inally', 'gether', ' ow', 'udden',
        'bye', 'ember', 'uddenly',
    ],
    'ARTICLE_A': [
        ' a',
    ],
    'THEY_I_WE_YOU': [
        ' They', ' they', ' I', 'I', ' we', ' You', 'They', ' We', 'You', 'We', 'Why', 'ople',
    ],
    'FRAGMENTS_1': [
        'er', 'e', 'a', 'le', 'ar', ' me', 'al', ' car', 'or', 'en', 'el', 'ce', 'um', 'nd',
        'ble', ' sweet', ' sand', ' fire', 'ace', ' spot', ' swim', 'ber', ' air', 'ee', 'ash',
        ' fur', 'der', 'eter', 'her', 'ower', ' pain', 'ia', ' win', ' nap', ' basket', ' free',
        'ree', 'cess', 'ara', 'hat', 'ren',
    ],
    'WITH_OF_FOR_MAKE': [
        ' with', ' of', ' for', ' make', ' made', ' took', ' take', ' grabbed', ' sees', ' making',
        ' enjoyed', ' makes', ' taking',
    ],
    'ONSETS_1': [
        ' c', ' b', ' s', ' t', ' sh', ' st', ' sp', ' d', ' g', ' ch', ' B', ' n', ' v', ' sc',
        ' L', ' P', ' te', ' pe', 'J', 'B', ' tr', ' th', ' R', ' j', ' J', ' sn', ' z', ' fe',
        ' qu', ' sw', ' y', ' G', ' sk', ' str', 'P', ' int', 'qu', 'T', ' squ', ' imp', ' N',
        'R', ' att', 'pl', ' ste', ' ph', ' cont', 'Z', ' ag', ' ne', ' beh', ' repl', ' thr',
        ' amaz', ' unt', ' qui',
    ],
    'CONJUNCTIONS': [
        ' that', ' but', ' what', ' But', ' as', ' when', 'But', ' if', ' because', ' until',
        ' And', 'When', ' When', ' where', ' As', ' why', 'As', ' wished', ' Maybe', ' sometimes',
        'ever', 'Maybe', 'where', ' wish', ' though', 'aybe', ';', 'fore',
    ],
    'VERB_ENDINGS': [
        'ing', 'ed', ' up', ' down', 've', 'led', 'king', 'ted', 'hing', 'ked', 'ome', ' fly',
        'ched', 'ke', ' waved', 'hed', 'ving', 'aring', 'ered', 'ase', 'ined', 'ting', 'ced',
        'kes', ' sing', 'ated', ' pulled', 'ling', ' ride', ' hide', 'ased', ' turn', 'ipped',
        'ming', 'de', 'pping', 'pped', ' sang', 'ning', ' moved', 'ocked', 'rew', 'ized', 'ied',
        'lew', 'ained', 'ther', 'iced',
    ],
    'ADJECTIVES_AND_QUANTIFIERS': [
        ' some', ' other', ' an', ' one', ' good', ' more', ' special', ' toy', ' nice', ' many',
        ' kind', 'ful', ' loud', ' long', ' beautiful', ' bad', ' pretty', ' shiny', ' great',
        ' bright', ' different', ' warm', 'ood', ' yummy', ' soft', ' funny', ' cool', ' own',
        ' delicious', ' hot', ' broken', ' enough', 'ious', ' few', ' scary', ' amazing', ' perfect',
        'icy', ' deep', ' strange', ' silly', 'thy', ' colorful', 'iting', ' gentle', ' magic',
        'ild', 'ible', 'cy', ' secret', ' year', ' naughty', ' wise', 'isy', '0', '5', ' blank',
        'ecial',
    ],
    'PLURALS_AND_OBJECT_PRONOUNS': [
        's', ' them', ' him', 'es', ' things', ' now', ' before', ' stopped', ' us', ' right',
        ' high', 'ies', ' world', ' doing', 'hes', ' move', 'ise', 'ings', 'gs', ' race', 'ze',
        'ins', ' turns', ' singing', 'And', 'ises', 'reed',
    ],
    'ONSETS_2': [
        ' p', ' m', ' f', ' l', 'r', ' r', ' w', ' M', 'M', ' k', 'v', ' cl', ' fl', ' wh', 'L',
        ' fr', ' gr', ' gl', ' kn', ' D', 'G', ' sl', ' pl', ' H', ' br', ' pr', ' bl', ' cr',
        ' de', ' bre', 'D', ' tw', ' K', ' ju', ' wr', 'H', ' pol', ' gu', ' rem', ' dec', 'N',
        ' spr', ' Fl', 'ract', ' sho', ' af', 'sp', ' rel', 'Y', ' underst', 'av', 'K', ' Sp',
        ' Th', ' welc', ' acc', 'Th',
    ],
    'FRAGMENTS_2': [
        'et', 'on', 'ad', 'ch', 'k', 'ent', 'est', 'ow', 'ver', 'ip', 'our', ' hand', 'ick',
        'ap', 'ight', 'ion', 'ep', ' rain', ' rock', 'own', 'ark', 'ack', ' mess', 'ess', ' rest',
        'ather', ' hair', 'nt', ' sleep', ' dream', 'ter', ' tight', 'ain', 'ast', 'all', 'red',
        'amp', ' plan', ' music', 'ort', 'fe', ' pot', 'ool', 'ear', 'uck', ' bath', 'be', 'ord',
        'nder', ' happen', ' shop', 'mer', 'ush', 'V', ' wood', 'wn', 'out', 'oy', 'X', 'arden',
    ],
    'MOTION_VERBS': [
        ' go', ' went', ' looked', ' ran', ' got', ' came', ' look', ' come', ' walked', ' flew',
        ' run', ' walking', ' looking', ' stay', ' fell', ' jumped', ' looks', ' coming', ' hopped',
        ' running', ' sat', ' arrived', ' far', ' stayed', ' turned', ' climbed', ' flying',
        ' runs', ' stepped', ' drove', 'ared', ' seemed', ' swam', ' woke',
    ],
    'HAD_SAW_FOUND': [
        ' had', ' saw', ' upon', ' have', ' found', ' heard', ' has', 'After', ' having', ' After',
        ' met', ' such', 'ith',
    ],
    'WORD_ENDINGS': [
        'y', 'ly', 'se', 'ay', 'ry', 'ine', 'ate', ' soon', 'ake', 'ite', 'ge', 'ant', 'ue',
        'ous', 'able', 'ish', 'ard', ' strong', ' first', 'zy', 'ide', ' dark', 'are', 'ave',
        'day', 'ven', 'ound', 'ath', 'dy', ' baby', 'ich', 'oth', ' cream', 'uffy', 'ty', 'ucky',
        'ary', ' paint', ' quiet', 'less', 'ire', ' heavy', ' suddenly', ' dry', ' apple', 'bby',
        'ail', 'ough', ' fair', 'ike', 'way', 'lly', 'ull', 'ady', 'ime', 'other', ' quick',
        'me', 'ix', '4', 'kay', '<0xA9>', '8', '3',
    ],
    'PREPOSITIONS': [
        ' in', ' on', ' from', ' into', ' under', ' through', ' near', ' behind', ' shared',
        ' across', ' eating', 'On', 'llow',
    ],
    'HAPPY_BACK_EXCITED': [
        ' happy', ' back', ' excited', ' sorry', ' way', ' hard', ' going', ' wait', ' listen',
        ' important', ' mean', ' goodbye', ' place', ' closer', ' thing', ' ready', ' himself',
        'ater', ' surprise', ' lesson', ' surprised', ' herself', ' close', ' agreed', ' talk',
        ' waited', ' money', ' worried', ' grateful', ' cheered', ' stories', 'fused', 'ward',
        'llo', 'sed',
    ],
    'CREATURES': [
        ' girl', ' boy', ' bird', ' man', ' dog', 'ie', ' bear', ' cat', ' rabbit', ' bunny',
        ' brother', ' lady', ' Joe', ' duck', ' mouse', ' frog', ' wind', ' bug', ' squirrel',
        ' lion', 'ob', ' fox', ' owner', ' princess', ' monkey', ' farmer', ' bee', ' doctor',
        ' fairy', ' child', ' elephant', ' owl', ' cow', ' kid', 'rot', 'key', 'cket', 'phant',
        'hen',
    ],
    'FRAGMENTS_3': [
        't', 'p', 'h', 'm', 'b', 'w', 'd', 'ill', 'll', 'am', 'z', 'ell', 'and', ' cut', ' hop',
        'os', 'ol', 'up', 'ff', ' cook', 'vel', 'ign', ' whe', ' past', ' kne', ' tow', 'uit',
        ' fin', 'ax', ' clos', 'med', ' flo', ' ab', ' ey', ' sec', ' pi', 'ong', 'lc', 'ott',
        ' ben', 'imb', ' purp', ' somet', ' favor', ' ke', ' lon', '<0xB1>', 'q',
    ],
    'FRAGMENTS_4': [
        'l', 'g', 'c', 'f', 'n', 'i', 'ir', 're', 'ur', 'ic', ' un', 'ist', '-', 'ach', 'he',
        'sh', ' al', 'iss', 'bb', ' comp', 'ang', 'omet', ' del', ' min', ' forg', 'urt', 'col',
        'F', ' cre', 'ht', ' delic', 'Q', ' contin', 'itc', '<0x9D>', ' ye', '(', 'hn', 'ourn',
        ' wal', '<', '<0x84>', '<0x98>',
    ],
    'FRIENDS_AND_GROUPS': [
        ' friends', ' friend', ' animals', 'ers', ' family', ' people', 'one', 'ice', ' truck',
        ' fish', ' voice', ' boat', ' doll', ' birds', ' kids', ' flower', 'ane', ' happily',
        ' butterfly', ' children', 'lf', 'ens', ' else', 'ey', ' others', 'man', ' puppy', ' sister',
        'cle', ' train', ' monster', ' dragon', ' tower', ' town', 'ier', ' animal', 'ade', 'ely',
        ' star', 'ppy', 'ger', 'ence', 'ame', 'ouse', 'gon', 'fully', 'nts', 'ner', ')', 'outh',
        'gry', ']',
    ],
    'REACTIONS_AND_FEELINGS': [
        ' smiled', ' sad', ' scared', ' careful', ' laughed', ' brave', ' angry', ' fast', ' nodded',
        ' eyes', ' hands', ' curious', ' tired', ' cars', ' cried', ' trees', ' cold', 'ons',
        ' colors', ' yes', ' moment', ' hungry', ' wet', ' smiles', ' faster', ' bigger', ' smart',
        ' dolls', ' dirty', ' crying', ' swings', ' higher', ' pocket', ' wings', ' shoes', ' upset',
        ' lonely', ' rocks', 'ds', 'ened', 'fter',
    ],
    'AROUND_HELP_INSIDE': [
        ' around', ' help', ' inside', ' off', ' by', ' hurt', ' hug', 'ive', ' share', ' explore',
        'ves', ' watched', ' stop', 'ved', 'ars', ' lost', 'ged', ' left', ' finished', 'ore',
        ' asks', ' visit', ' leave', 'side', 'ored', ' bite', 'ank', ' clapped', 'orrow', ' along',
        'ches', ' exploring', ' drink', 'akes', 'ited', 'ange', 'asure', 'bbed', 'ished', 'ued',
    ],
    'OBJECTS': [
        ' ball', ' box', ' end', 'ck', ' bed', ' walk', ' head', ' work', ' stick', ' dress',
        'ile', 'ure', ' jump', 'ak', ' game', ' hat', 'oon', 'ient', ' light', 'ation', ' window',
        ' bag', ' swing', ' laugh', ' wall', 'orn', 'ance', ' fall', ' pictures', ' balloon',
        ' field', ' pieces', ' cup', ' blanket', ' treasure', ' sign', ' mouth', ' bowl', ' heart',
        ' Mr', ' chair', 'ors', 'ead', ' key', 'ountain', 'ek', ' jar', 'bow', ' bucket', 'adow',
        ' bow', 'etter', 'ook', ' shout', 'arn', ' slow', ' color', 'ream', 'ask', ' block',
        'row', 'rel', 'ield', 'ture',
    ],
    'FRAGMENTS_VC': [
        'in', 'at', 'an', 'ot', 'it', 'il', 'ig', 'op', 'as', 'un', 'ut', 'is', 'om', 'us', 'x',
        'im', 'iz', ' bus', 'ug', 'ct', 'em', 'art', ' ex', 'ster', 'ng', 'Jo', 'oug',
    ],
    'VERBS_BEFORE_TO': [
        ' wanted', ' loved', ' started', ' want', ' liked', ' decided', ' tried', ' began', ' likes',
        ' able', ' continued', ' wants', ' trying', ' listened', ' promised', ' explained',
    ],
    'IT_AND_EVERYONE': [
        ' it', ' everyone', ' story', ' name', ' everything', 'chool', 'ello',
    ],
    'ONSETS_3': [
        'u', ' co', ' o', ' che', ' sa', ' ro', ' mo', 'O', ' se', ' la', ' E', ' ru', ' po',
        'ro', ' tri', ' gra', ' bu', ' app', 'bo', ' ho', ' cra', ' bo', ' wor', ' mu', ' tra',
        'if', ' tal', ' spark', 'udd', ' pat', 'ugg', 'igg', ' pa', ' bott', ' cho', ' ve', ' bar',
        ' em', ' disapp', ' je', 'ere', ' pu', ' gre', ' su', ' relie', ' ca', ' stu', 'au',
        ' Gra', ' cu', ' dif', ' perf', ' butterf', ' buck', ' pare', ' belie', ' mon', ' el',
        'so', ' dro', 'fere',
    ],
    'BARE_VERBS_TRANSITIVE': [
        ' see', ' put', ' find', ' get', ' about', ' eat', ' keep', ' after', ' use', ' open',
        ' ate', ' touch', ' catch', ' used', ' held', ' buy', ' hear', ' brought', ' wear', ' reach',
        ' All', ' join', ' threw', ' believe', ' without', ' hit', 'ought', ' bring', ' caught',
        'ken', 'dded', 'ided',
    ],
    'QUOTE_ENDS': [
        '."', '!"', '?"', ' named', '?', '}',
    ],
    'MENTAL_VERBS': [
        ' like', ' thought', ' knew', ' how', ' over', ' learned', ' From', ' try', ' need',
        ' noticed', ' sure', ' glad', ' realized', ' than', ' seen', ' think', ' -', ' remembered',
        ' needed', ' forgot', 'From', ' ask', ' thinks', ' remember', ' hoped', ' forget', 'ised',
        'ause',
    ],
    'SO_VERY_REALLY': [
        ' so', ' very', ' feel', ' really', ' feels',
    ],
    'VERBS_BEFORE_A_PERSON': [
        ' at', ' asked', ' told', ' gave', ' hugged', ' thanked', ' called', ' opened', ' showed',
        ' helped', ' followed', 'At', ' towards', ' helping', ' closed', 'cked', ' pushed', 'issed',
        ' hugs', ' shook', ' dropped', 'ught', 'apped', 'ired', 'ached',
    ],
    'FRAGMENTS_5': [
        'o', ' sun', 'ra', 'oo', ' A', ' lo', 'A', 'th', "'", ' ice', ' ha', 'co', ' snow', ' fo',
        ' To', ' won', 'ink', ' Jen', ' pen', 'zz', ' heal', ' accident', 'ho', 'umb', ' neck',
        ' Jo', ' lea', ' ba', 'sel', 'lp', 'ma', ' clo', ' sto', ' cast', '2', 'urp', '<0xA6>',
        '*', '_', 'asket', '<0xB0>', '<0xB3>', '>', '[', '<0xA8>', '<0xBD>', '<0x92>', '<0x99>',
        '<0xBA>',
    ],
    'FIXED_CONTINUATIONS': [
        'Once', 'â', ' didn', ' couldn', ' don', ' â', 'Don', ' dri', ' ta', ' wasn', ' wouldn',
        ' fore', ' prom', ' Em', 'eci', '<0x80>', '<0xE2><0x82><0xAC><0xE2>', ' hun', ' vis',
        ' stre', ' arri', '<0xCB>', '<0xC2>', '<0xC3>', '<0xE2>', '<0xAD>', '<0xC5>', '<0xAC>',
        '<0xB8>', '<0xAA>', '<0x86>', '<0x85>', '<0x84><0xA2>', '<0xAE>', '<0xC6>', '<0x91>',
    ],
    'FRAGMENTS_6': [
        'id', 'ag', 'ust', 'ld', 'itt', ' pretend', 'ind', 'pp', ' tast', ' smell', 'ss', ' hid',
        'ump', ' count', 'ass', ' fight', 'urn', 'oom', ' wonder', 'ress', ' sail', ' roll',
        ' land', 'nn', ' answ', ' scream', ' pass', ' mix', ' pack', ' stand', ' corn', ' plac',
        'pt', ' exp', 'eet', ' mail', ' yell', ' happ', ' smil', 'uff', ' gent', 'irl', 'elt',
        'irst', 'app', ' op', ' fi', 'ctor', 'unch', ' beaut', ' nam', ' pap', 'ount', ' shap',
        ' flow', 'maz', 'sw', 'aught',
    ],
    'SIZE_AND_COLOUR': [
        ' big', ' old', ' small', ' red', ' blue', ' green', ' yellow', ' tall', ' friendly',
        ' pink', ' white', ' real', ' teddy', ' black', ' brown', 'iny', ' young', ' nearby',
        ' purple', ' orange', 'avy',
    ],
    'ONSETS_4': [
        ' re', ' li', '€™', 'ri', ' com', ' dis', ' en', 'E', ' ra', ' creat', ' exc', ' fa',
        'uc', ' poin', 'ou', ' mar', ' dan', ' sear', ' ad', ' pre', ' con', 'ac', ' bran', ' gif',
        ' pro', ' prin', ' u', ' pus', ' ar', ' trou', ' laug', 'oun', ' mat', ' es', ' des',
        'ret', ' war', ' bir', ' stor', ' bra', ' ac', ' sor', ' dir', ' pic', ' per', 'U', ' mag',
        ' fam', ' din', ' bel', ' ma', ' loo', ' bro', ' tre', 'fu', ' fro', ' ele', 'aut', ' differe',
        ' pret', ' mus', ' wat', ' curi', ' hig',
    ],
    'THERE_AND_IT': [
        ' there', ' It', ' Everyone', ' There', ' Jim', 'thing',
    ],
    'TOO_DO_SOMETHING': [
        ' too', ' do', ' something', ' much', ' even', ' ever', ' bit', ' anything', ' someone',
        ' most', ' longer', ' less',
    ],
    'OUT_PROUD_LOTS': [
        ' out', ' proud', ' lots', ' lot', ' sound', ' care', ' top', ' full', ' picture', ' arm',
        ' song', ' treat', ' cloud', 'air', ' moral', ' side', ' snack', ' plant', ' twins',
        'bble', ' carrot', ' fruit', 'ts', ' piece', ' part', 'oup', 'igh', 'riend', ' list',
        'imal', 'eces', '<0xB2>',
    ],
    'ALL_EVERY_ONE': [
        ' all', ' One', 'One', ' every', ' Every', ' next', 'very', ' 3',
    ],
    'END_OF_STORY_AND_W': [
        '<|eos|>', ' wa', ' W', 'W', ' jo', ' coll', ' journ', ' inst', ' tom', ' resp', 'gan',
        ' wo', 'j', ' dra', ' vo', ' surpr', '1', 'urpr', 'rom', '9', '<0xA3>', '<0x93>', '<0xB6>',
    ],
    'ONSETS_5': [
        ' h', ' e', ' S', 'S', ' T', ' le', ' C', ' F', ' gi', 'ul', ' dr', ' spl', 'C', ' cle',
        ' cray', ' im', ' butt', 'br', ' neighb', ' pers', ' smo', ' dist', ' inv', ' col', ' expl',
        ' bec', ' advent', ' fav', ' ide', ' sm', ' import', '|',
    ],
    'PLAY_VERBS': [
        ' play', ' playing', ' played', ' lived', ' worked', ' filled', ' sit', ' stuck', 'itting',
    ],
    'YOU_AND_PLEASE': [
        ' you', ' please', 'Oh', 'lease', 'hy', 'itchen',
    ],
    'BARE_VERBS': [
        ' know', ' love', 'ect', ' watch', ' learn', ' draw', ' fix', ' hold', ' read', ' understand',
        ' build', 'atter', ' start', ' break', ' enjoy', 'ft', ' call', ' miss', ' belong', 'iver',
        'dd', 'ross', 'vent', 'joy',
    ],
    'BLANK_LINES_AND_BYTES': [
        ' \n', '€', '<0x83>', '<0x82>', '<0x9A>', '<0x9E>', '<0xBB>', '<0xA4>', '~', '<0xB7>',
        '<0x89>', '<0xB5>',
    ],
    'PHRASE_ENDINGS_RARE': [
        ' lake', 'ope', ' party', ' dance', ' river', ' candy', ' bike', ' dangerous', ' words',
        'eared', ' school', ' today', ' games', ' yard', 'ment', ' noises', ' late', ' adventures',
        ' well', ' lunch', ' sea', ' street', ' mine', ' fence', ' soup', ' stars', ' backyard',
        'ces', ' farm', 'ions', ' shapes', ' bench', 'ity', 'self', 'oney', 'leep', 'ards', 'oring',
        ' butter', ' pie', ' danger', 'set', 'ister', 'uch', 'yard', 'more',
    ],
    'SHOW_GIVE_PICK': [
        ' show', ' clean', ' picked', ' give', ' pick', ' tell', ' reached', ' climb', ' set',
        ' broke', ' pull', 'ried', ' follow', ' throw', 'ock', ' fill',
    ],
    'NO_ANY_THREE': [
        ' no', 'st', ' any', ' three', ' In', 'end', ' pet', ' grow', 'orm', 'fort', 'get', 'xt',
        'ec', 'ph',
    ],
    'IT_LET_WHAT_UNSPACED': [
        'It', ' let', 'Let', 'What', 'That', ' Let', ' That', ' What', 'This',
    ],
    'THIS_AND_FAVORITE': [
        ' this', ' favorite', ' another', ' wonderful', ' This', ' living', ' magical', ' same',
        'aughty',
    ],
    'THANK_CAN_DO': [
        'Thank', 'Can', ' Do', ' Can', ' thank', 'Do', ' Thank', 'ould',
    ],
    'NAMES_AND_FAMILY_RARE': [
        ' teacher', 'here', ' Mary', 'Lila', ' Grandma', ' Emma', 'aughter', 'by', 'andy', 'eddy',
        'ucy', 'aisy', 'unny', 'ames', 'ndma', 'bbit', 'ila',
    ],
    'ONSETS_4_RARE': [
        ' teac', ' squir', ' cour', ' beauti', ' grate', '/', ' clot', ' mor', '<0xB4>',
    ],
    'OBJECTS_RARE': [
        'ident', 'oud', ' neigh',
    ],
    'WORD_ENDINGS_RARE': [
        '6', '7', '+', '%',
    ],
    'NEVER_IN_TRAINING_TEXT_1': [
        '^', '<0xC0>', '<0xC1>', '<0xC4>', '<0xC7>', '<0xC8>', '<0xC9>', '<0xCA>', '<0xCC>',
        '<0xCD>', '<0xCE>', '<0xCF>', '<0xD0>', '<0xD1>', '<0xD2>', '<0xD3>', '<0xD4>', '<0xD5>',
        '<0xD6>', '<0xD7>', '<0xD8>', '<0xD9>', '<0xDA>', '<0xDB>', '<0xDC>', '<0xDD>', '<0xDE>',
        '<0xDF>', '<0xE0>', '<0xE1>', '<0xE3>', '<0xE4>', '<0xE5>', '<0xE6>', '<0xE7>', '<0xE8>',
        '<0xE9>', '<0xEA>', '<0xEB>', '<0xEC>', '<0xED>', '<0xEE>', '<0xEF>', '<0xF0>', '<0xF1>',
        '<0xF2>', '<0xF3>', '<0xF4>', '<0xF5>', '<0xF6>', '<0xF7>', '<0xF8>', '<0xF9>', '<0xFA>',
        '<0xFB>', '<0xFC>', '<0xFD>', '<0xFE>', '<0xFF>', '\x00', '\x01', '\x02', '\x03', '\x04',
    ],
    'NEVER_IN_TRAINING_TEXT_2': [
        '\x05', '\x06', '\x07', '\x08', '\x0b', '\x0c', '\r', '\x0e', '\x0f', '\x10', '\x11',
        '\x12', '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a', '\x1b', '\x1c',
        '\x1d', '\x1e', '\x1f', '\x7f', '<0x81>', '<0x87>', '<0x88>', '<0x8B>', '<0x8C>', '<0x8D>',
        '<0x8E>', '<0x8F>', '<0x90>', '<0x94>', '<0x95>', '<0x96>', '<0x97>', '<0x9B>', '<0x9F>',
        '<0xE2><0x82>',
    ],
}

TARGET_CLASSES = {
    'SENTENCE_ENDS': [
        '.', '!', '."', '!"', '?"', ',"', ' again', '?', ' anymore', '".', ' -', 'ment', ' fore',
        'sel', ';', ' somet', ']', 'uddenly', '}',
    ],
    'DETERMINERS_AND_OBJECT_PRONOUNS': [
        ' the', ' her', ' his', ' them', ' him', ' their', ' me', ' each', ' its', ' us', ' everyone',
        ' our', ' himself', ' herself', ' others', ' helping', ' school', ' turns', ' danger',
        ' amaz',
    ],
    'SENTENCE_STARTS': [
        ' She', ' He', ' They', ' The', '<|eos|>', ' \n', ' It', ' One', ' But', ' You', ' We',
        ' And', ' So', ' Her', ' Then', ' His', ' From', ' Every', ' When', ' Suddenly', ' Let',
        ' Do', ' As', ' Everyone', ' Can', ' That', ' In', ' After', ' Now', ' There', ' Maybe',
        ' What', ' This', ' To', ' All', ' Thank',
    ],
    'PAST_VERBS': [
        ' had', ' saw', ' wanted', ' went', ' loved', ' could', ' looked', ' would', ' did',
        ' ran', ' started', ' made', ' found', ' liked', ' got', ' decided', ' didn', ' took',
        ' thought', ' knew', ' tried', ' couldn', ' heard', ' played', ' learned', ' lived',
        ' walked', ' flew', ' kept', ' opened', ' picked', ' stopped', ' noticed', ' quickly',
        ' became', ' fell', ' jumped', ' helped', ' sees', ' reached', ' realized', ' ate', ' began',
        ' used', ' hopped', ' worked', ' wished', ' likes', ' sat', ' ta', ' remembered', ' held',
        ' needed', ' enjoyed', ' followed', ' arrived', ' forgot', ' continued', ' wants', ' stayed',
        ' turned', ' climbed', ' met', ' listened',
    ],
    'CONJUNCTIONS': [
        ' and', ' as', ' when', ' if', ' because', ' or', ' where', ' than', ' today', ' inst',
    ],
    'TO_THAT_FOR': [
        ' to', ' that', ' for', ' about', ' before', ' ever', ' enough',
    ],
    'ONSETS_1': [
        ' p', ' c', ' m', ' b', ' s', ' t', ' f', ' sh', ' st', ' car', ' sp', ' l', ' d', ' r',
        ' h', ' g', ' w', ' re', ' ch', ' e', ' n', ' k', ' co', ' v', ' li', ' hug', ' sc',
        ' red', ' wa', ' che', ' te', ' sa', ' cl', ' lo', ' un', ' ro', ' fl', ' le', ' pe',
        ' walk', ' tr', ' blue', ' rain', ' th', ' stick', ' mo', ' wh', ' sn', ' bus', ' se',
        ' z', ' la', ' fr', ' ru', ' po', ' gr', ' gl', ' com', ' tri', ' kn', ' mess', ' rest',
        ' gra', ' cut',
    ],
    'LINE_BREAKS_AND_QUOTE_BYTES': [
        '\n', 'â', 'œ', "'", '<0xCB>', '<0xC2>', '`', '\\', '<0xC5>', '\t', '~', '{',
    ],
    'COMMAS': [
        ',', ':', ' goodbye', ' yes',
    ],
    'NAMES': [
        ' Lily', ' I', ' Ben', ' Tom', ' Timmy', ' ', ' Tim', ' Sam', ' Max', ' Mom', ' Anna',
        ' M', ' Jack', ' B', ' S', ' Mia', ' A', ' L', ' Sara', ' P', ' T', ' Lucy', ' W', ' Spot',
        ' Mommy', ' C', ' Bob', ' Sue', ' F', ' R', ' J', ' E', ' Sarah', ' Jane', ' Dad', ' Joe',
        ' D', ' John', ' G', ' H', ' Sally', ' Daisy', ' Lila', ' Benny', ' Tommy', ' K', ' Billy',
        ' Amy', ' Jen', ' Jim', ' Daddy', ' N', ' Molly', ' Fl', ' Mr', ' Jill', ' Mary', ' Jo',
        ' Grandma', ' Emma', ' Em', ' Gra', ' Sp', ' Th',
    ],
    'BARE_VERBS_1': [
        ' play', ' be', ' go', ' help', ' see', ' make', ' do', ' find', ' get', ' know', ' take',
        ' eat', ' come', ' feel', ' show', ' try', ' share', ' keep', ' explore', ' run', ' give',
        ' listen', ' work', ' stay', ' use', ' fly', ' learn', ' cry', ' draw', ' worry', ' touch',
        ' catch', ' climb', ' buy', ' move', ' hear', ' understand', ' build', ' ask', ' talk',
        ' sing', ' sit', ' ride', ' visit', ' leave', ' hide', ' wear', ' dance', ' reach', ' join',
        ' win', ' turn', ' believe', ' bring', ' drink', 'gan', ' fin', ' perf', ' beh', ' contin',
        ' arri',
    ],
    'WAS_AND_IS': [
        ' was', "'s", ' is', 'my', ' has', ' does', ' looks', ' makes', ' wasn', ' seemed', ' teac',
    ],
    'A_AND_AN': [
        ' a', ' an', ' many', ' lots', ' been', ' sure', ' done', ' such',
    ],
    'IT_YOU_SOMETHING': [
        ' it', ' you', ' some', ' what', ' something', ' your', ' one', ' my', ' how', ' this',
        ' being', ' any', ' happened', ' why', ' another', ' anything', ' someone', ' everything',
        'joy',
    ],
    'PREPOSITIONS': [
        ' in', ' with', ' on', ' together', ' from', ' outside', ' until', ' by', ' every', ' after',
        ' here', ' under', ' near', ' behind', ' happily', ' without', ' games', ' ab', ' ben',
    ],
    'SPEECH_AND_REACTION_VERBS': [
        ' said', ' smiled', ' asked', ' felt', ' says', ' put', ' came', ' told', ' gave', ' hugged',
        ' thanked', ' laughed', ' called', ' nodded', ' replied', ' showed', ' watched', ' grabbed',
        ' shouted', ' waved', ' cried', ' agreed', ' smiles', ' asks', ' promised', ' scream',
        ' stepped', ' clapped', ' explained', ' hugs', ' shook', ' woke', ' yell', ' wat', '/',
    ],
    'SUBJECT_PRONOUNS': [
        ' he', ' she', ' they', ' we', ' then', ' thank', ' sometimes', 'ever', ' suddenly',
        'leep',
    ],
    'FEELINGS_AND_QUALITIES': [
        ' happy', ' sad', ' scared', ' good', ' sorry', ' nice', ' kind', ' better', ' careful',
        ' hard', ' bad', ' brave', ' angry', ' pretty', ' shiny', ' important', ' bright', ' strong',
        ' warm', ' curious', ' sweet', ' glad', ' soft', ' tired', ' funny', ' cool', ' dark',
        ' surprised', ' cold', ' delicious', ' friendly', ' hot', ' scary', ' hungry', ' wet',
        ' pat', ' int', ' amazed', ' silly', ' con', ' cle', ' quiet', ' worried', ' grateful',
        ' imp', ' smart', ' dangerous', ' em', ' dirty', ' gentle', ' free', ' lonely', ' naughty',
        ' relie', ' slow', ' advent', ' dif', ' delic', ' quick', ' thr', ' qui', ' ke',
    ],
    'SUFFIXES': [
        'ing', 'ed', 'er', 'y', 'es', 'ly', 'ers', 'ge', 'ion', '-', 'bo', 'ered', 'ens', 'room',
        'der', 'ier', 'ible', 'bow', 'ions', 'ened', 'ually', 'reed', 'hn', 'yard', 'more', '>',
    ],
    'FRAGMENTS_1': [
        'e', 'a', 'o', 'in', 'at', 'et', 'an', 'i', 'ie', 'en', 'el', 'ic', 'am', 'age', 'ish',
        'ro', 'ies', 'ation', 'end', 'ined', 'all', 'ries', 'urn', 'her', 'ty', 'ort', 'ol',
        'ased', 'ire', 'ade', 'mer', 'eet', 'adow', 'ity', 'ards', 'ma', 'ime', 'eci', 'ired',
        'ished', 'other', 'ong', 'oy', 'ount', 'hen',
    ],
    'OPENERS_UNSPACED': [
        'The', 'I', 'They', 'But', 'He', 'Yes', 'It', 'You', 'S', 'When', 'No', 'Let', 'Mom',
        'What', 'That', 'Thank', 'Suddenly', 'She', 'We', 'So', 'Look', 'W', 'O', 'Her', 'Can',
        'After', 'Wow', 'G', 'As', 'Timmy', 'Don', 'Hello', 'E', 'Then', 'From', 'OK', 'His',
        'At', 'This', 'Oh', 'Okay', 'Maybe', 'Sure', 'Why', 'Hi', 'Finally', 'On', 'Do', 'And',
        'Y', 'Z', 'U', 'Q', 'Th', '$', '&', '[', '@', '|', '<0xC6>',
    ],
    'TIME_AND_SIZE_AFTER_A': [
        ' time', ' little', ' big', ' new', ' special', ' small', ' loud', ' long', ' beautiful',
        ' lot', ' while', ' great', ' bit', ' tall', ' picture', ' wonderful', ' moment', ' few',
        ' deep', ' gif', ' magical', ' nap', ' magic', ' young', ' secret', ' nearby', ' wise',
        ' bench', ' pic', ' fam', 'dded', ' fi', 'nts', ' blank',
    ],
    'SO_VERY_NOT': [
        ' so', ' very', ' not', ' too', ' no', ' safe', ' feeling', ' okay', ' right', ' three',
        ' ready', ' full', ' wrong', ' able', ' gone', ' twins', ' stuck', ' welcome', ' mine',
        ' af',
    ],
    'FRAGMENTS_2': [
        'le', 'l', 'b', 'f', 'w', 'd', 'se', 'st', 're', 'll', 'led', 'nd', 'les', 'ck', 'king',
        'ked', 'x', 'ny', 'ved', 'zy', 'pp', 'sh', 'ct', 'ient', 'nt', 'ss', 'ven', 'ber', 'dy',
        'bb', ' cream', 'man', 'ling', 'zz', 'orm', 'thy', 'nn', 'ff', 'ins', 'eared', 'de',
        'bble', 'cked', 'pping', 'pped', 'be', 'bby', 'ppy', 'col', 'sp', 'by', 'dd', 'gether',
        'j', 'get', 'wn', 'rel', 'bbed', 'sed', 'ndma', 'set', 'uch', 'llow', 'kay',
    ],
    'FIXED_CONTINUATIONS': [
        "'t", 'Once', ' upon', 'g', 'h', '€', 've', 'ver', '€™', 'ves', 'ld', 'ect', 'ke', 'ving',
        'day', 'ally', 'lf', 'ise', 'ey', 'kes', 'ily', 'ith', 'orrow', 'nder', 'here', 'ho',
        'ead', 'hy', 'ng', 'ized', 'ised', 'ited', 'lly', 'ken', 'ught', 'ould', 'gon', '<0x80>',
        '<0xE2><0x82><0xAC><0xE2>', 'ises', '<0x9C>', 'lc', '<0x9D>', '0', '<0xC3>', '<0xA6>',
        '<0x83>', 'nna', '<0xA9>', '<0x82>', '<0xA0>', '<0xE2>', '<0xAD>', '<0xB4>', '<0xB1>',
        '<0xB9>', '<0xA2>', '<0xB0>', '<0xB3>', '<0x9A>', '<0xA1>', '<0xAC>', '<0xAF>', '<0xA5>',
    ],
    'FRAGMENTS_3': [
        't', 'p', 'm', 'c', 'n', 'ch', 'k', 'ful', 'ce', 'ry', 'v', 'ted', 'ous', 'hing', 'ble',
        'ched', 'z', 'ure', 'he', 'th', 'hed', 'hes', 'ter', 'co', 'ting', 'ious', 'ced', 'os',
        'fe', 'cle', 'ze', 'fort', 'ft', 'ried', 'ming', 'iting', ' course', 'br', 'ts', 'pl',
        'cy', 'vel', 'ning', 'ches', 'very', 'pt', 'fused', 'ward', 'ces', 'ger', 'cess', 'ht',
        'row', 'xt', 'asure', 'rot', 'vent', 'ctor', 'fully', 'ther', 'fu', 'fore', 'cket', 'ph',
    ],
    'PARTICLES': [
        ' up', ' at', ' back', ' around', ' out', ' away', ' home', ' down', ' into', ' inside',
        ' over', ' off', ' closer', ' through', ' towards', ' across', ' everywhere', ' along',
    ],
    'S_OF_NAMED': [
        's', ' of', ' named', ' who', 'less',
    ],
    'ANIMALS_AND_OBJECTS': [
        ' man', ' dog', ' tree', ' ball', ' box', ' bear', ' cat', ' truck', ' fish', ' slide',
        ' rock', ' boat', ' j', ' cake', ' sand', ' fire', ' story', ' lady', ' duck', ' flower',
        ' frog', ' hole', ' pl', ' snow', ' mu', ' baby', ' bug', ' balloon', ' field', ' cloud',
        ' party', ' pot', ' bran', ' bott', ' cup', ' train', ' princess', ' monkey', ' bowl',
        ' bee', ' pu', ' bath', ' chair', ' town', ' butt', ' animal', ' piece', ' yard', ' corn',
        ' apple', ' key', ' elephant', ' star', ' shop', ' jar', ' owl', ' whe', ' ca', ' farm',
        ' cow', ' stor', ' block', ' mag', ' din',
    ],
    'FRAGMENTS_4': [
        'ad', 'id', 'ot', 'it', 'ay', 'ow', 'il', 'ra', 'is', 'ice', 'ist', 'ite', 'om', 'itt',
        'own', 'ark', 'ood', 'ile', 'ind', 'ater', 'ide', 'ane', 'ound', 'ath', 'atter', 'ich',
        'ump', 'iss', 'ama', 'if', 'ass', 'ummy', 'udd', 'uffy', 'ink', 'ucky', 'ang', 'ower',
        'ope', 'ool', 'olly', 'issed', 'ild', 'ract', 'ountain', 'ocked', 'ush', 'ough', 'etter',
        'ike', 'ock', 'ucy', 'ax', 'aybe', 'irl', 'oup', 'elt', 'apped', 'av', 'aisy', 'ame',
        'app', 'ouse', 'unch',
    ],
    'QUOTE_THERE_BUT': [
        ' "', ' there', ' but', ' â', ' please', ' ow',
    ],
    'FRAGMENTS_5': [
        'ent', 'est', 'ill', 'ig', 'un', 'oo', 'ip', 'ick', 'ake', 'ust', 'im', 'iz', 'ug', 'ace',
        'ack', 'ach', 'ell', 'ather', 'and', 'em', 'are', 'ave', 'ase', 'ore', 'ain', 'uc', 'ou',
        'icy', 'amp', 'oom', 'ress', 'eter', 'ob', 'ape', 'ugg', 'igg', 'lease', 'up', 'ac',
        'ipped', 'uck', 'ere', 'illy', 'ord', 'umb', 'itting', 'iny', 'oun', 'ought', 'ail',
        'aughter', 'urt', 'ret', 'arn', 'rew', 'ied', 'au', 'uit', 'ree', 'uff', 'ross', 'ream',
        'ittle', 'ull',
    ],
    'FAMILY_AND_BODY': [
        ' mom', ' friends', ' friend', ' mommy', ' dad', ' hand', ' eyes', ' hands', ' head',
        ' face', ' mum', ' favorite', ' name', ' parents', ' own', ' arm', ' mother', ' grandma',
        ' daddy', ' mouth', ' pocket', 'ster', 'self', ' bir', ' kne', ' fav', ' nam', ' pare',
    ],
    'NAMES_UNSPACED': [
        '"', 'Lily', 'One', 'Tom', 'M', 'Ben', 'J', 'B', 'A', 'Anna', 'Sara', 'L', 'Mommy', 'Tim',
        'John', 'Sam', 'Jack', 'D', 'P', 'H', 'C', 'Max', 'T', 'N', 'R', 'Mia', 'Lila', 'ign',
        'F', 'Jo', 'K', '1', '2', 'X', '5', '4', '9', '8', '3', '6', '7', '(', '*', '=', '<',
        '#',
    ],
    'HAVE_CAN_LIKE': [
        ' have', ' can', ' like', ' want', ' love', ' don', ' say', ' both', ' should', ' need',
        "'re", ' think', ' hop', ' must', ' forg', ' wish', ' prom',
    ],
    'BARE_VERBS_2': [
        ' always', ' look', ' hurt', ' let', ' clean', ' wait', ' watch', ' mean', ' pick', ' tell',
        ' open', ' stop', ' jump', ' care', ' pretend', ' fix', ' hold', ' read', ' swim', ' sleep',
        ' set', ' laugh', ' pr', ' start', ' break', ' poin', ' fight', ' fall', ' wr', ' spl',
        ' grow', ' roll', ' enjoy', ' rem', ' dec', ' pull', ' call', ' remember', ' follow',
        ' stand', ' pus', ' forget', ' happen', ' throw', ' inv', ' es', ' resp', ' shout', ' cre',
        ' expl', ' bec', ' fill', ' tre', ' repl', ' unt', ' squir', ' neigh', ' ye',
    ],
    'ONSETS_2': [
        ' bu', ' qu', ' app', ' tast', ' sw', ' light', ' green', ' ho', ' gi', ' dri', ' cra',
        ' yellow', ' dis', ' y', ' swing', ' bo', ' jo', ' wor', ' dream', ' sk', ' br', ' en',
        ' ha', ' ra', ' broken', ' bl', ' cr', ' de', ' tra', ' str', ' creat', ' count', ' pink',
        ' tw', ' fo', ' ju', ' perfect', ' comp', ' dr', ' white', ' pol', ' mar', ' real', ' pet',
        ' spark', ' sail', ' ad', ' ex', ' plan', ' pre', ' land', ' pain', ' gu', ' pa', ' paint',
        ' race', ' answ', ' colorful', ' squ', ' cho', ' pro', ' plant', ' black', ' sign',
    ],
    'ADVERBS': [
        ' all', ' never', ' just', ' even', ' still', ' soon', ' now', ' also', ' first', ' only',
        ' really', ' sl', ' smell', ' carefully', ' lost', ' left', ' al', ' finished', ' close',
        ' hid', ' might', ' tight', ' filled', ' finally', ' tal', ' alone', ' dan', ' sear',
        ' slowly', ' faster', ' disapp', ' del', ' heal', ' higher', ' hit', ' belong', ' gre',
        ' caught', ' gent', ' sor', ' ag', ' ne', ' flo', ' welc', ' sm', ' shap', ' flow', ' grate',
    ],
    'FRAGMENTS_6': [
        'ar', 'on', 'or', 'ur', 'op', 'um', 'as', 'ag', 'our', 'ant', 'ome', 'ars', 'oon', 'ons',
        'art', 'ee', 'ance', 'oth', 'ash', 'ated', 'ia', 'omet', 'ored', 'ank', 'ventually',
        'oney', 'inally', 'ook', 'V', 'andy', 'eddy', 'med', 'unny', 'riend', 'urpr', 'ila',
        'outh',
    ],
    'PLACES': [
        ' park', ' other', ' sun', ' end', ' garden', ' sky', ' door', ' ground', ' store', ' kitchen',
        ' grass', ' world', ' top', ' floor', ' window', ' air', ' wind', ' table', ' wall',
        ' beach', ' woods', ' living', ' moral', ' most', ' same', ' swings', ' sea', ' street',
        ' ele', ' import', ' hig',
    ],
    'CREATURES_AND_NATURE': [
        ' girl', ' boy', ' bird', ' forest', ' rabbit', ' bunny', ' pond', ' mouse', ' butterfly',
        ' hill', ' squirrel', ' lion', ' puppy', ' fox', ' lake', ' monster', ' river', ' dragon',
        ' farmer', ' doctor', ' fairy', ' child', ' fence',
    ],
    'WERE_ARE_WILL': [
        ' were', ' are', ' will', "'m", ' am', "'ll",
    ],
    'MASS_AND_PLURAL_NOUNS': [
        ' things', ' noise', ' food', ' flowers', ' sound', ' cars', ' blocks', ' paper', ' ice',
        ' books', ' colors', ' bre', ' leaves', ' cookies', ' dinner', ' pictures', ' joy', ' music',
        ' side', ' money', ' pieces', ' candy', ' ve', ' treasure', ' words', ' carrot', ' fruit',
        ' noises', ' stories', ' adventures', ' lunch', ' soup', ' rocks', ' shapes', ' past',
        ' tow', ' col', ' wood', ' butter', ' bro', ' butterf', ' pap', ' surpr', ' mon', ' ide',
        ' purp', ' clot',
    ],
    'FRAGMENTS_7': [
        'u', 'r', 'ine', 'ate', 'ut', 'ive', 'ight', 'ue', 'us', 'ess', 'ul', 'ged', 'ast', 'orn',
        'red', 'qu', 'ary', 'gs', 'imes', 'ors', 'ely', 'iver', 'akes', 'ady', 'ara', 'ence',
        'ued', 'key', 'ren', 'ways', 'ident', 'rom', 'imb', 'oud', 'chool', 'imal', 'eces', 'gry',
        'q',
    ],
    'BELONGINGS': [
        ' toy', ' house', ' room', ' way', ' family', ' voice', ' doll', ' book', ' game', ' hat',
        ' surprise', ' lesson', ' bag', ' castle', ' teddy', ' pen', ' owner', ' blanket', ' bike',
        ' tower', ' basket', ' teacher', ' neighb', ' journ', ' neck', ' backyard', ' ph', ' bucket',
        ' bra', ' clos', ' beaut', ' list', ' favor',
    ],
    'TIMES_OF_DAY': [
        ' day', ' night', ' morning', 'where', ' year', 'thing',
    ],
    'FRAGMENTS_8': [
        'al', 'ir', 'one', 'ap', 'ep', 'able', 'ard', 'ri', 'ak', ' years', 'aring', 'ings',
        'air', 'ear', 'side', 'ared', ' longer', 'ek', 'way', 'ds', 'isy', 'igh', 'ause', 'out',
        'ached', 'itc',
    ],
    'PLURALS_AND_OLD': [
        ' old', ' water', ' best', ' o', ' animals', ' two', ' people', ' next', ' birds', ' kids',
        ' trees', ' children', ' stars', ' orange', ' dir', ' sto', ' op', ' pret', ' el',
    ],
    'FUN_MORE_DIFFERENT': [
        ' fun', ' more', ' different', ' yummy', ' seen', ' else', ' exc', ' far', ' amazing',
        ' strange', ' bigger', ' heavy', ' late', ' trou', ' exp', ' though', ' happ', 'lp',
        ' less',
    ],
    'ING_FORMS': [
        ' playing', ' going', ' walking', ' looking', ' having', ' making', ' coming', ' doing',
        ' running', ' getting', ' trying', ' flying', ' crying', ' eating', ' singing', ' laug',
        ' exploring', ' taking', ' smil', ' dro',
    ],
    'ONSETS_3': [
        ' bar', ' je', ' prin', ' cook', ' pass', ' bite', ' spr', ' att', ' brown', ' mix',
        ' coll', ' im', ' pack', ' u', ' ste', ' pers', ' smo', ' dry', ' sho', ' ar', ' miss',
        ' dist', ' mat', ' su', ' 3', ' des', ' purple', ' lea', ' mail', ' stu', ' part', ' bow',
        ' war', ' rel', ' ba', ' fair', ' ac', ' cont', ' wo', ' per', ' clo', ' dra', ' loo',
        ' sec', ' cast', ' fro', ' buck', ' cour', ' wal',
    ],
    'EXCITED_MUCH_PROUD': [
        ' excited', ' much', ' proud', ' fast', ' high', ' well', ' upset', 'fter', ' beauti',
    ],
    'BODY_AND_BELONGINGS': [
        ' toys', ' bed', ' fe', ' brother', ' hair', ' fa', ' fur', ' clothes', ' tail', ' sister',
        ' dolls', ' heart', ' cray', ' wings', ' shoes', ' ey', ' stre', ' mus', ' acc', ' curi',
        ' lon',
    ],
    'IDEA_SMILE_PLACE': [
        ' idea', ' smile', ' place', ' dress', ' adventure', ' thing', ' spot', ' song', ' treat',
        ' job', ' snack', ' min', ' plac', ' tom', ' color', ' bel', 'bye', '%',
    ],
    'PAST_VERBS_RARE': [
        ' wonder', ' brought', ' pulled', ' won', ' waited', ' shared', ' runs', ' thinks', ' broke',
        ' threw', ' wouldn', ' closed', ' hoped', ' feels', ' drove', ' pushed', ' cheered',
        ' accident', ' sang', ' moved', ' dropped', 'llo', ' swam', ' underst', ' hun', ' vis',
        ' belie',
    ],
    'FRAGMENTS_5_RARE': [
        'lew', 'ask', 'oring', 'irst', 'ange', 'ield', 'ained', 'ames', 'ec', 'udden', 'ott',
        'hat', 'ix', 'aut', 'avy', 'oug', 'urp', 'ourn', 'aught', 'arden', 'asket', 'ello',
    ],
    'ANIMALS_AND_OBJECTS_RARE': [
        ' cu', ' ma', ' pie', ' kid', ' vo', ' pi', ' differe', ' mor',
    ],
    'FRAGMENTS_3_RARE': [
        'me', 'ture', 'ople', 'ner', 'bbit', 'ember', 'so', 'phant',
    ],
    'FRAGMENTS_4_RARE': [
        'iced', 'ister', 'ided', 'ecial', 'aughty', 'itchen',
    ],
    'FRAGMENTS_2_RARE': [
        'maz', 'fere', 'sw', '_', '+',
    ],
    'CLOSING_BRACKET': [
        ')',
    ],
    'FIXED_CONTINUATIONS_RARE': [
        '<0xBF>', '<0xA7>', '<0xA8>', '<0xBD>', '<0x84>', '<0x9E>', '<0xBB>', '<0xA4>', '<0xA3>',
        '<0xB8>', '<0xBE>', '<0xAA>', '<0x86>', '<0x92>', '<0x99>', '<0xBA>', '<0x85>', '<0xAB>',
        '<0x93>', '<0xB7>', '<0xBC>', '<0xB6>', '<0x8A>', '<0x84><0xA2>', '<0x98>', '<0xAE>',
        '<0xB2>', '<0x89>', '<0xB5>', '<0x91>',
    ],
    'NEVER_IN_TRAINING_TEXT_1': [
        '^', '<0xC0>', '<0xC1>', '<0xC4>', '<0xC7>', '<0xC8>', '<0xC9>', '<0xCA>', '<0xCC>',
        '<0xCD>', '<0xCE>', '<0xCF>', '<0xD0>', '<0xD1>', '<0xD2>', '<0xD3>', '<0xD4>', '<0xD5>',
        '<0xD6>', '<0xD7>', '<0xD8>', '<0xD9>', '<0xDA>', '<0xDB>', '<0xDC>', '<0xDD>', '<0xDE>',
        '<0xDF>', '<0xE0>', '<0xE1>', '<0xE3>', '<0xE4>', '<0xE5>', '<0xE6>', '<0xE7>', '<0xE8>',
        '<0xE9>', '<0xEA>', '<0xEB>', '<0xEC>', '<0xED>', '<0xEE>', '<0xEF>', '<0xF0>', '<0xF1>',
        '<0xF2>', '<0xF3>', '<0xF4>', '<0xF5>', '<0xF6>', '<0xF7>', '<0xF8>', '<0xF9>', '<0xFA>',
        '<0xFB>', '<0xFC>', '<0xFD>', '<0xFE>', '<0xFF>', '\x00', '\x01', '\x02', '\x03', '\x04',
    ],
    'NEVER_IN_TRAINING_TEXT_2': [
        '\x05', '\x06', '\x07', '\x08', '\x0b', '\x0c', '\r', '\x0e', '\x0f', '\x10', '\x11',
        '\x12', '\x13', '\x14', '\x15', '\x16', '\x17', '\x18', '\x19', '\x1a', '\x1b', '\x1c',
        '\x1d', '\x1e', '\x1f', '\x7f', '<0x81>', '<0x87>', '<0x88>', '<0x8B>', '<0x8C>', '<0x8D>',
        '<0x8E>', '<0x8F>', '<0x90>', '<0x94>', '<0x95>', '<0x96>', '<0x97>', '<0x9B>', '<0x9F>',
        '<0xE2><0x82>',
    ],
}

CLASS_SYSTEMS = {"ATTENTION": ATTENTION_CLASSES, "CONTEXT": CONTEXT_CLASSES, "TARGET": TARGET_CLASSES}
