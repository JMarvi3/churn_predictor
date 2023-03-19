$(document).ready(function(){
    let pie = null
    $("#PhoneService").change()
    $("#InternetService").change()
    let submit = function() {
        let form = $("form");
        $.ajax({
            type: "POST",
            url: form.attr("action"),
            data: form.serialize(),
            success: function (response) {
                // "rgb(255, 99, 132)","rgb(54, 162, 235)"
                // "rgb(230, 240, 249)", "rgb(8, 48, 107)"
                // "rgba(230, 240, 249, .7)", "rgba(8, 48, 107, .7)"
                // "rgb(204, 230, 253)", "rgb(76, 106, 151)"
                Chart.defaults.global.defaultFontStyle = 'bold'
                Chart.defaults.global.defaultFontColor = 'black'
                Chart.defaults.global.defaultFontSize = 20
                let data = {
                    datasets: [{
                        data: response["probabilities"],
                        backgroundColor: ["rgb(204, 230, 253)", "rgb(76, 106, 151)"]
                    }], labels: ['No', 'Yes']
                }
                if (pie) pie.destroy()
                pie = new Chart($("#pieChart"), {
                    type: 'pie', data: data, options: {
                        responsive: true,
                        plugins: {
                            datalabels: {
                                formatter: function (value, context) {
                                    return context.chart.data.labels[context.dataIndex] + ': ' + Math.round(value) + '%';
                                }
                            }
                        },
                        tooltips: {enabled: false},
                        legend: {display: false},
                        title: {display: true, text: response['prediction']}
                    }
                })
            }
        });
    }
    $("input").on('change', function(event) {
        if (isNaN(this.value)) {
            this.setCustomValidity('Please enter a number.')
        } else if (this.value<0) {
            this.setCustomValidity('Please enter a number that is no less than 0.')
        } else {
            this.setCustomValidity('')
            if ($(':invalid').length===0) {
                submit()
            }
        }
    })
    $("select").on('change', submit)
    submit()
})

function phoneChange(phone) {
    let needs = $(".needsPhoneService")
    if (phone.value === "No") {
        needs.find('option[value!="No phone service"]').prop('hidden', true)
        needs.find('option[value="No phone service"]').prop('hidden', false)
        needs.find("select").val("No phone service")
        needs.children().addBack().css('color','grey')
    } else {
        needs.find('option[value!="No phone service"]').prop('hidden', false)
        needs.find('option[value="No phone service"]').prop('hidden', true)
        needs.find("select").val("No")
        needs.children().addBack().css('color','')
    }
}
function internetChange(internet) {
    let needs = $(".needsInternet")
    if (internet.value === "No") {
        needs.find('option[value!="No internet service"]').prop('hidden', true)
        needs.find('option[value="No internet service"]').prop('hidden', false)
        needs.find("select").val("No internet service")
        needs.children().addBack().css('color','grey')
    } else if (internet.oldvalue === "No") {
        needs.find('option[value!="No internet service"]').prop('hidden', false)
        needs.find('option[value="No internet service"]').prop('hidden', true)
        needs.find("select").val("No")
        needs.children().addBack().css('color','')
    }
    internet.oldvalue = internet.value;
}