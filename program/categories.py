"""Word classes: every token belongs to exactly one named class.

Found by Ward clustering of each token's left and right neighbours in the TRAIN split
(analysis/taxonomy.py; no model involved), then named by reading the members
(analysis/gen_categories.py). Each class has at most 64 tokens so that per-token tables
can be written one class at a time. Members are listed most frequent first.
"""

from .vocab import ID

CLASSES = {
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

CLASS_OF = {ID[t]: name for name, toks in CLASSES.items() for t in toks}
CLASS_NAMES = list(CLASSES)
