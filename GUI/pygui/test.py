
<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Bokeh Plot</title>
        
<link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-0.12.5.min.css" type="text/css" />
<link rel="stylesheet" href="https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.12.5.min.css" type="text/css" />
        
<script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-0.12.5.min.js"></script>
<script type="text/javascript" src="https://cdn.pydata.org/bokeh/release/bokeh-widgets-0.12.5.min.js"></script>
<script type="text/javascript">
    Bokeh.set_log_level("info");
</script>
        <style>
          html {
            width: 100%;
            height: 100%;
          }
          body {
            width: 90%;
            height: 100%;
            margin: auto;
          }
        </style>
    </head>
    <body>
        
        <div class="bk-root">
            <div class="bk-plotdiv" id="6cfa12e2-629e-47f9-9d47-e2776be0e714"></div>
        </div>
        
        <script type="text/javascript">
            (function() {
          var fn = function() {
            Bokeh.safely(function() {
              var docs_json = {"d368b807-0dc6-417c-a69e-214d172337a8":{"roots":{"references":[{"attributes":{"callback":null,"options":["1","2","3"],"title":"Second","value":"1"},"id":"964396c1-2343-4ac8-87aa-3c14767e093d","type":"Select"},{"attributes":{"callback":null,"icon":null,"label":"Dropdown button","menu":[["Item 1","item_1"],["Item 2","item_2"],null,["Item 3","item_3"]]},"id":"ea2f3029-81c5-4400-b57c-cae1fef7aa36","type":"Dropdown"},{"attributes":{"callback":null,"options":["A","B"],"title":"First","value":"A"},"id":"976a3fa6-a176-4c1a-aea4-3172099ef31a","type":"Select"},{"attributes":{"children":[{"id":"1319b535-a03e-44fa-b051-ec92080426c6","type":"Select"},{"id":"00229e31-4b6c-470f-937a-3a843897729f","type":"Select"}]},"id":"04961658-62c1-49ee-abff-8bd2a16ac60e","type":"WidgetBox"},{"attributes":{"callback":null,"options":["A","B"],"title":"First","value":"A"},"id":"97e4ab04-85c5-4aab-b5bf-a1aedf18dc50","type":"Select"},{"attributes":{"callback":null,"icon":null,"label":"Dropdown button","menu":[["Item 1","item_1"],["Item 2","item_2"],null,["Item 3","item_3"]]},"id":"d5e795e7-fe4b-4653-acc6-51a7a9b435fc","type":"Dropdown"},{"attributes":{"children":[{"id":"d58d4b72-ebe6-4fec-ac0e-7cb764a40d30","type":"Select"},{"id":"1159e814-5b71-4ea8-b0f6-71c187eb5242","type":"Select"}]},"id":"7b73d2a0-7361-4109-b521-f86fed8a7e87","type":"WidgetBox"},{"attributes":{"children":[{"id":"aeee8259-3898-4cd9-984a-31146af93c8e","type":"Select"},{"id":"964396c1-2343-4ac8-87aa-3c14767e093d","type":"Select"}]},"id":"0cef4929-120f-4ea2-b7dd-0a7afc75ac61","type":"WidgetBox"},{"attributes":{"callback":null,"options":["1","2","3"],"title":"Second","value":"1"},"id":"528d6d2b-f2e6-44f0-9a41-fa29a6d06fce","type":"Select"},{"attributes":{"callback":null,"options":["A","B"],"title":"First","value":"A"},"id":"d58d4b72-ebe6-4fec-ac0e-7cb764a40d30","type":"Select"},{"attributes":{"button_type":"warning","callback":null,"icon":null,"label":"Dropdown button","menu":[["Item 1","item_1"],["Item 2","item_2"],null,["Item 3","item_3"]]},"id":"f3d9f715-859f-4c7e-b0a9-d9d587b828ac","type":"Dropdown"},{"attributes":{"callback":null,"options":["1","2","3"],"title":"Second","value":"1"},"id":"554c4e83-6180-4a6d-b3ae-cc738807d7a1","type":"Select"},{"attributes":{"children":[{"id":"976a3fa6-a176-4c1a-aea4-3172099ef31a","type":"Select"},{"id":"554c4e83-6180-4a6d-b3ae-cc738807d7a1","type":"Select"}]},"id":"82203b5b-0b20-440c-9e2a-452ca0c5d983","type":"WidgetBox"},{"attributes":{"children":[{"id":"97e4ab04-85c5-4aab-b5bf-a1aedf18dc50","type":"Select"},{"id":"528d6d2b-f2e6-44f0-9a41-fa29a6d06fce","type":"Select"}]},"id":"0d27856c-c4c5-4b74-9307-c7f6460f5355","type":"WidgetBox"},{"attributes":{"callback":null,"options":["A","B"],"title":"First","value":"A"},"id":"aeee8259-3898-4cd9-984a-31146af93c8e","type":"Select"},{"attributes":{"callback":null,"options":["1","2","3"],"title":"Second","value":"1"},"id":"00229e31-4b6c-470f-937a-3a843897729f","type":"Select"},{"attributes":{"callback":null,"options":["1","2","3"],"title":"Second","value":"1"},"id":"1159e814-5b71-4ea8-b0f6-71c187eb5242","type":"Select"},{"attributes":{"callback":null,"options":["A","B"],"title":"First","value":"A"},"id":"1319b535-a03e-44fa-b051-ec92080426c6","type":"Select"}],"root_ids":["f3d9f715-859f-4c7e-b0a9-d9d587b828ac","ea2f3029-81c5-4400-b57c-cae1fef7aa36","0d27856c-c4c5-4b74-9307-c7f6460f5355","0cef4929-120f-4ea2-b7dd-0a7afc75ac61","d5e795e7-fe4b-4653-acc6-51a7a9b435fc","04961658-62c1-49ee-abff-8bd2a16ac60e","82203b5b-0b20-440c-9e2a-452ca0c5d983","7b73d2a0-7361-4109-b521-f86fed8a7e87"]},"title":"Bokeh Application","version":"0.12.5"}};
              var render_items = [{"docid":"d368b807-0dc6-417c-a69e-214d172337a8","elementid":"6cfa12e2-629e-47f9-9d47-e2776be0e714","modelid":"7b73d2a0-7361-4109-b521-f86fed8a7e87"}];
              
              Bokeh.embed.embed_items(docs_json, render_items);
            });
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
        
        </script>
    </body>
</html>