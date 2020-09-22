var index = {
    build: function (allData) {
        var dataList;
        if (typeof type != "undefined" && type == 'all') {
            var listKeys = Object.keys(allData);
            listKeys.sort();
            listKeys.forEach(function (key){
                var symbolData = allData[key];
                var symbol = key;
                dataList = symbolData;
                index.buildSymbolBlock(dataList, symbol);
            })
        } else {
            dataList = allData;
            index.buildSymbolBlock(dataList);
        }

    },
    buildSymbolBlock: function (dataList, symbol) {
        var rootsection = document.getElementById('rootsection');
        var currentSection = rootsection.cloneNode(true);
        rootsection.parentElement.appendChild(currentSection);
        currentSection.classList.remove('is-hidden');
        if (symbol) {
            var title = currentSection.querySelector('.title');
            title.setAttribute('id', symbol);
            title.innerHTML = symbol;
        }
        var entries = Object.entries(dataList);
        entries.forEach(function (dataItem) {
            var baseContainer = currentSection.querySelector('.data-container');
            var dataContainer = baseContainer.cloneNode(true);
            baseContainer.parentElement.appendChild(dataContainer);
            dataContainer.querySelector('.subtitle').innerHTML = 'Epoch ' + dataItem[0];
            index.buildSection(dataContainer, dataItem);
        })
    },
    buildSection: function (currentSection, dataItem) {
        currentSection.classList.remove('is-hidden');
        var data = dataItem[1];
        index.buildTable(currentSection, data);
        index.buildStatistic(currentSection, data);
    },
    buildStatistic: function (currentSection, data) {
        var container = currentSection.querySelector('.statistic-container');
        var fisrtChild = container.querySelector('.column');
        fisrtChild.remove();
        var entries = Object.entries(data);
        entries.forEach(function (entry) {
            var item = fisrtChild.cloneNode(true);

            if (entry[0].includes('max_moving_average_') || entry[0].includes('max_delay_session_')) {
                item.innerHTML = entry[0].replace('max_moving_average_', 'MA ').replace('max_delay_session_', 't+') + ' lớn nhất: ' + entry[1];
                container.appendChild(item);
            }

            if (entry[0].includes('max_value_matrix')) {
                item.classList.remove('is-4')
                item.innerHTML = `<b>Max</b>: <b>${entry[1]['value']}</b> tại <b>t+${entry[1]['delay_session']}</b> và <b>MA ${entry[1]['moving_average']}</b> `
                container.appendChild(item);
            }

        })
    },
    buildTable: function (currentSection, data) {
        var table = currentSection.querySelector('table');
        var thead = table.querySelector('thead');
        var header = document.createElement('tr');
        header.innerHTML = '<th></th>';
        data.predict_delay_session_list.forEach(function (delay_session) {
            var th = document.createElement('th');
            th.innerHTML = 't+' + delay_session;
            header.appendChild(th);
        })
        thead.appendChild(header);

        var tbody = table.querySelector('tbody');
        data.value_matrix.forEach(function (rowData, rowIndex) {
            var tr = document.createElement('tr');
            tr.innerHTML = `<td>MA ${(rowIndex + 1) * 10}</td>`;
            rowData.forEach(function (cellItem) {
                var td = document.createElement('td');
                td.innerHTML = cellItem;
                tr.appendChild(td);
            })
            tbody.appendChild(tr);
        })

    }
};